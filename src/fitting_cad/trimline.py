"""Rule-based trim-curve engine.

Roadmap: Week 6 (rule engine), Week 7 (validation).

Generate the socket trim line (rim curve) from anatomical landmarks using
clinical convention, rather than learned/ML methods. The curve is what a
prosthetist would otherwise draw by hand on every socket.

Canonical frame (set by :func:`fitting_cad.io.normalize_orientation`):

* limb long axis -> **+Z**, proximal rim at ``Z = 0`` (wide end),
* limb tapers toward **+Z**, origin at the proximal-rim centroid.

**Anterior direction.** This engine fixes **+X as anterior** in the canonical
frame (so -X is posterior, and +/-Y are the medial/lateral sides). The
normalization step pins the long axis and origin but does *not* yet pin the
rotation about Z, so this is a convention the interactive/clinical layer must
ultimately enforce (e.g. by rotating the frame so the patellar tendon sits at
+X). A prosthetist should confirm the anterior axis before the curve is trusted
for a real socket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from fitting_cad.landmarks import LandmarkSet


@dataclass
class TrimRules:
    """Tunable clinical parameters for trim-line generation.

    Defaults are placeholders -- calibrate against clinician-reviewed examples.

    Attributes:
        anterior_offset_mm: How much **lower** (toward the rim, ``-Z``) the trim
            dips anteriorly, over the patellar tendon, to relieve the patella.
            Larger -> more anterior relief.
        posterior_height_mm: How much **higher** (toward the distal end, ``+Z``)
            the trim rises posteriorly for the popliteal trim. Larger -> a higher
            posterior wall.
        smoothing: Loop smoothing strength in ``[0, 1]``. ``0`` leaves the
            ray-sampled loop as-is; higher values apply more angular smoothing of
            the height profile for a cleaner rim.
    """

    anterior_offset_mm: float = 15.0  # anterior relief depth (placeholder)
    posterior_height_mm: float = 10.0  # popliteal trim rise (placeholder)
    smoothing: float = 0.3  # height-profile smoothing in [0, 1]

    extra: dict = field(default_factory=dict)


# Anterior is +X in the canonical frame (documented in the module docstring).
_ANTERIOR = np.array([1.0, 0.0, 0.0])


def generate_trimline(
    mesh: trimesh.Trimesh,
    landmarks: "LandmarkSet",
    rules: TrimRules | None = None,
    resolution: int = 180,
) -> np.ndarray:
    """Return an ordered, closed loop of on-surface points (the trim curve).

    First-pass geometric rule engine. The loop wraps the limb once and lies on
    the mesh surface, so it can be lofted into a socket rim or exported.

    Algorithm (all in the canonical frame; clinical parameters in ``rules`` are
    explicit placeholders to be calibrated later):

    1. **Base height.** Anchor a base trim height ``z0`` off the landmark
       Z-heights -- the mean Z of the placed required landmarks. This puts the
       rim in the right band down the limb without hard-coding a number.
    2. **Angular height profile.** Sweep angles ``theta in [0, 2*pi)``. Build a
       smooth target height ``z(theta)`` that dips anteriorly (near +X, the
       patellar-tendon region) by ``anterior_offset_mm`` and rises posteriorly
       (near -X, the popliteal) by ``posterior_height_mm``. The dip/rise use a
       ``cos(theta)`` weighting so the transition through the medial/lateral
       walls (+/-Y) is smooth.
    3. **Optional smoothing.** Blend the height profile toward its angular mean
       by ``smoothing`` for a cleaner rim.
    4. **Project onto the surface.** For each ``theta`` at height ``z(theta)``,
       cast a ray from the limb axis (``x = y = 0`` at that ``z``) outward along
       ``(cos theta, sin theta, 0)`` and take the farthest mesh-surface hit (the
       outer wall). Misses are dropped and back-filled by interpolation from
       their angular neighbours.
    5. **Close + resample.** Order by ``theta``, close the loop, and resample to
       a uniform ``resolution`` points.

    Args:
        mesh: A watertight limb mesh in the canonical frame.
        landmarks: Placed landmarks; the required subset anchors the base height.
        rules: Clinical parameters. Defaults to :class:`TrimRules` placeholders.
        resolution: Number of points in the returned loop.

    Returns:
        An ``(resolution, 3)`` float64 array: an ordered, closed, on-surface loop.

    Raises:
        ValueError: Landmarks are missing, ``resolution`` is too small, or no
            surface intersections could be found (degenerate mesh/placement).
    """
    if rules is None:
        rules = TrimRules()
    if resolution < 8:
        raise ValueError(f"resolution must be >= 8, got {resolution}.")

    missing = landmarks.missing()
    if missing:
        names = ", ".join(lm.value for lm in missing)
        raise ValueError(f"Cannot generate trim line; missing landmarks: {names}.")

    # --- 1. Base height from the landmark Z-band -----------------------------
    landmark_z = np.array(
        [landmarks.points[lm][2] for lm in landmarks.points], dtype=np.float64
    )
    z0 = float(landmark_z.mean())

    # --- 2. Angular height profile -------------------------------------------
    theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    # cos(theta): +1 anterior (+X), -1 posterior (-X), 0 at medial/lateral.
    anterior_weight = np.cos(theta)
    # Lower anteriorly (toward the rim, -Z), higher posteriorly (toward +Z).
    z_profile = (
        z0
        - rules.anterior_offset_mm * np.clip(anterior_weight, 0.0, None)
        + rules.posterior_height_mm * np.clip(-anterior_weight, 0.0, None)
    )

    # --- 3. Optional angular smoothing of the height profile -----------------
    s = float(np.clip(rules.smoothing, 0.0, 1.0))
    if s > 0.0:
        z_profile = (1.0 - s) * z_profile + s * float(z_profile.mean())

    # --- 4. Project each (theta, z) onto the outer surface via a ray ---------
    points = _project_rays_to_surface(mesh, theta, z_profile)

    # --- 5. Close + resample to a uniform loop -------------------------------
    loop = _resample_closed_loop(points, resolution)
    return loop


def _project_rays_to_surface(
    mesh: trimesh.Trimesh, theta: np.ndarray, z_profile: np.ndarray
) -> np.ndarray:
    """Ray-cast outward from the axis at each (theta, z); take the outer hit.

    Returns an ``(N, 3)`` array ordered by ``theta``. Angles whose ray misses
    the surface are back-filled by interpolating the radius from their nearest
    angular neighbours that did hit, so the loop stays continuous.
    """
    n = len(theta)
    directions = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
    origins = np.column_stack([np.zeros(n), np.zeros(n), z_profile])

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins, ray_directions=directions, multiple_hits=True
    )

    # For each ray, keep the farthest hit (outer wall). Track radius for back-fill.
    radius = np.full(n, np.nan)
    hit_points = np.full((n, 3), np.nan)
    for ray_i in range(n):
        mask = index_ray == ray_i
        if not np.any(mask):
            continue
        hits = locations[mask]
        r = np.linalg.norm(hits[:, :2] - origins[ray_i, :2], axis=1)
        far = int(np.argmax(r))
        radius[ray_i] = r[far]
        hit_points[ray_i] = hits[far]

    if np.all(np.isnan(radius)):
        raise ValueError(
            "No surface intersections found for the trim line; check that the "
            "mesh is in the canonical frame and the landmark band lies on it."
        )

    # Back-fill missed rays by interpolating radius over theta (periodically),
    # then reconstruct the point on the cylinder-of-revolution at that radius.
    valid = ~np.isnan(radius)
    if not np.all(valid):
        # Periodic interpolation: extend theta/radius across the 2*pi seam.
        tv = theta[valid]
        rv = radius[valid]
        ext_theta = np.concatenate([tv - 2.0 * np.pi, tv, tv + 2.0 * np.pi])
        ext_r = np.concatenate([rv, rv, rv])
        order = np.argsort(ext_theta)
        filled_r = np.interp(theta, ext_theta[order], ext_r[order])
        for ray_i in np.where(~valid)[0]:
            r = filled_r[ray_i]
            hit_points[ray_i] = [
                r * np.cos(theta[ray_i]),
                r * np.sin(theta[ray_i]),
                z_profile[ray_i],
            ]

    return hit_points


def _resample_closed_loop(points: np.ndarray, resolution: int) -> np.ndarray:
    """Resample an ordered open polyline into a uniform closed loop.

    Treats ``points`` as samples around a closed loop (already ordered by angle),
    closes it, and resamples to ``resolution`` points evenly spaced by arc length.
    """
    closed = np.vstack([points, points[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 0.0:
        raise ValueError("Degenerate trim loop (zero arc length).")

    # resolution points around the loop, excluding the duplicated closing point.
    targets = np.linspace(0.0, total, resolution, endpoint=False)
    out = np.empty((resolution, 3), dtype=np.float64)
    for axis in range(3):
        out[:, axis] = np.interp(targets, arc, closed[:, axis])
    return out


def validate_trimline(curve: np.ndarray, mesh: trimesh.Trimesh) -> dict:
    """Sanity-check a generated curve and report metrics.

    Roadmap: Week 7 (validation). No GUI.

    Metrics returned:

    * ``point_count``: number of points in the loop.
    * ``closure_gap``: distance from the last point back to the first
        (loop continuity; should be ~ the mean segment length for a uniform loop).
    * ``mean_segment_length`` / ``max_segment_length``: spacing diagnostics.
    * ``max_distance_to_surface``: farthest any loop point sits from the mesh
        surface (should be ~0 for an on-surface curve).
    * ``self_intersections``: count of crossing segment pairs in the XY
        projection (a planar proxy; a clean rim loop should be 0).

    Args:
        curve: An ``(N, 3)`` ordered, closed loop of points.
        mesh: The mesh the curve was generated on.

    Returns:
        A dict of the metrics above.

    Raises:
        ValueError: ``curve`` is not an ``(N, 3)`` array with ``N >= 3``.
    """
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 3 or curve.shape[0] < 3:
        raise ValueError(f"curve must be an (N, 3) array with N >= 3, got {curve.shape}.")

    n = curve.shape[0]

    # Closure + spacing (treat as a closed loop: last segment wraps to first).
    wrapped = np.vstack([curve, curve[0]])
    seg_len = np.linalg.norm(np.diff(wrapped, axis=0), axis=1)
    closure_gap = float(np.linalg.norm(curve[-1] - curve[0]))

    # Distance to surface for every loop point.
    closest, distance, _ = trimesh.proximity.closest_point(mesh, curve)
    max_dist = float(np.max(distance))

    # Self-intersection count in the XY projection.
    self_int = _count_xy_self_intersections(curve[:, :2])

    return {
        "point_count": int(n),
        "closure_gap": closure_gap,
        "mean_segment_length": float(seg_len.mean()),
        "max_segment_length": float(seg_len.max()),
        "max_distance_to_surface": max_dist,
        "self_intersections": int(self_int),
    }


def _count_xy_self_intersections(xy: np.ndarray) -> int:
    """Count crossing pairs of non-adjacent segments in a closed XY polyline."""
    n = xy.shape[0]
    pts = np.vstack([xy, xy[0]])  # close the loop
    count = 0
    for i in range(n):
        a1, a2 = pts[i], pts[i + 1]
        # Compare against later segments; skip adjacent ones (share an endpoint),
        # including the wrap-around adjacency between the last and first segment.
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # first and last segments are adjacent on a closed loop
            b1, b2 = pts[j], pts[j + 1]
            if _segments_cross(a1, a2, b1, b2):
                count += 1
    return count


def _segments_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    """True if open segments ``p1p2`` and ``p3p4`` properly intersect in 2D."""

    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    return (d1 * d2 < 0.0) and (d3 * d4 < 0.0)
