"""Anatomical landmark placement and the non-interactive plumbing around it.

Roadmap: Week 5 (landmark-placement step).

The clinician places ~6-8 anatomical landmarks on the normalized limb mesh.
These drive the rule-based trim-curve generation in :mod:`fitting_cad.trimline`.

This module keeps the *non-interactive* parts (the :class:`LandmarkSet`
container, a nearest-vertex snap helper, and the required-landmark report)
free of any GUI / GL dependency so they are testable headless. The interactive
picker (:func:`place_landmarks`) is a separate UI task and is still a stub.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from collections.abc import Mapping


class Landmark(str, Enum):
    """The transtibial landmark set a prosthetist references for the trim line."""

    PATELLA = "patella"
    PATELLAR_TENDON = "patellar_tendon"
    TIBIAL_TUBERCLE = "tibial_tubercle"
    FIBULAR_HEAD = "fibular_head"
    MEDIAL_CONDYLE = "medial_condyle"
    LATERAL_CONDYLE = "lateral_condyle"
    POPLITEAL = "popliteal"  # posterior reference


# Landmarks the first-pass trim-line rule engine needs to anchor its z(theta)
# profile. The patellar tendon anchors the anterior relief, the popliteal anchors
# the posterior (higher) trim, and the fibular head + condyles fix the
# medial/lateral wall heights. The patella and tibial tubercle are useful context
# for a clinician but are not strictly required by the current rule, so they are
# left out of the required set.
REQUIRED_LANDMARKS: tuple[Landmark, ...] = (
    Landmark.PATELLAR_TENDON,
    Landmark.FIBULAR_HEAD,
    Landmark.MEDIAL_CONDYLE,
    Landmark.LATERAL_CONDYLE,
    Landmark.POPLITEAL,
)


@dataclass
class LandmarkSet:
    """Resolved 3D positions for each placed landmark (model coordinates)."""

    points: dict[Landmark, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, points: "Mapping[Landmark, np.ndarray | list[float] | tuple[float, ...]]"
    ) -> LandmarkSet:
        """Build a :class:`LandmarkSet` from a mapping of landmark -> point.

        Each value is coerced to a ``(3,)`` float64 array. This is the
        non-interactive entry point used by tests and by callers that already
        have landmark coordinates (e.g. loaded from a persisted JSON sidecar).

        Args:
            points: Mapping from :class:`Landmark` to a 3D point.

        Returns:
            A :class:`LandmarkSet` with copied, validated points.

        Raises:
            ValueError: A value is not a 3-vector.
        """
        resolved: dict[Landmark, np.ndarray] = {}
        for landmark, point in points.items():
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
            if arr.shape != (3,):
                raise ValueError(
                    f"Landmark {landmark.value!r} must be a 3D point, got shape "
                    f"{np.asarray(point).shape}."
                )
            resolved[landmark] = arr
        return cls(points=resolved)

    def missing(self) -> list[Landmark]:
        """Required landmarks (see :data:`REQUIRED_LANDMARKS`) not yet placed.

        Returns:
            The required landmarks absent from :attr:`points`, in canonical order.
        """
        return [lm for lm in REQUIRED_LANDMARKS if lm not in self.points]


def snap_to_surface(mesh: trimesh.Trimesh, point: np.ndarray) -> np.ndarray:
    """Snap a 3D point to the nearest **vertex** of ``mesh``.

    A clinician's pick (or a programmatic landmark) rarely lands exactly on a
    mesh vertex; snapping to the nearest vertex keeps landmark positions
    reproducible and tied to concrete mesh geometry. Uses trimesh's vertex
    KD-tree (``mesh.kdtree``), so it is GUI-free and fast.

    Args:
        mesh: The mesh whose vertices are candidate snap targets.
        point: A 3D point in the same coordinate frame as ``mesh``.

    Returns:
        A ``(3,)`` float64 copy of the nearest mesh vertex.

    Raises:
        ValueError: ``point`` is not a 3-vector, or the mesh has no vertices.
    """
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"point must be a 3D point, got shape {np.asarray(point).shape}.")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        raise ValueError("Cannot snap to a mesh with no vertices.")

    # mesh.kdtree is a scipy cKDTree over the vertices; query the single nearest.
    _, index = mesh.kdtree.query(p)
    return vertices[int(index)].copy()


def load_landmarks_json(path: str | Path, mesh: trimesh.Trimesh | None = None) -> LandmarkSet:
    """Load a landmark sidecar (name -> [x, y, z]) into a :class:`LandmarkSet`.

    The sidecar is a JSON object mapping each landmark name to a 3-element list,
    e.g.::

        {
            "patellar_tendon": [12.0, 0.0, 40.0],
            "popliteal": [-30.0, 0.0, 55.0],
            ...
        }

    **Coordinate-frame convention.** Landmark coordinates are interpreted in the
    **canonical limb frame** -- i.e. as if placed on the *normalized* mesh
    returned by :func:`fitting_cad.io.normalize_orientation` (long axis ``+Z``,
    proximal rim at ``Z = 0``, origin at the proximal-rim centroid), **not** in
    raw scanner coordinates. Generate landmarks against the normalized mesh and
    persist them in that frame. This matters because the pipeline normalizes the
    mesh before generating the trim line, so a sidecar in scanner coordinates
    would not line up with the surface.

    Args:
        path: Path to the JSON sidecar.
        mesh: Optional **normalized** mesh. When given, each landmark is snapped
            to the nearest mesh vertex via :func:`snap_to_surface` for
            robustness, so small off-surface offsets in the sidecar are tied back
            to concrete geometry.

    Returns:
        A :class:`LandmarkSet` in the canonical frame.

    Raises:
        FileNotFoundError: The sidecar path does not exist.
        ValueError: The JSON is not an object, a key is not a known landmark
            name, or a value is not a 3-vector.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark sidecar not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"{path.name} must be a JSON object mapping landmark name -> [x, y, z], "
            f"got {type(raw).__name__}."
        )

    valid_names = {lm.value: lm for lm in Landmark}
    points: dict[Landmark, np.ndarray] = {}
    for name, value in raw.items():
        if name not in valid_names:
            known = ", ".join(sorted(valid_names))
            raise ValueError(f"Unknown landmark {name!r} in {path.name}; known names: {known}.")
        landmark = valid_names[name]
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        if mesh is not None:
            point = snap_to_surface(mesh, point)
        points[landmark] = point

    return LandmarkSet.from_dict(points)


def place_landmarks(mesh: trimesh.Trimesh) -> LandmarkSet:
    """Open the interactive picker and return the placed landmark set.

    Still a stub: this is the interactive landmark-placement UI task (#6),
    pending. It will open a pyvista point-picking session, snap each pick to the
    nearest mesh vertex via :func:`snap_to_surface`, label the landmarks, allow
    re-picking, and persist the set to a JSON sidecar beside the scan. The
    non-interactive plumbing it depends on (:class:`LandmarkSet`,
    :func:`snap_to_surface`) is implemented and testable headless.
    """
    raise NotImplementedError(
        "place_landmarks is the interactive landmark-picker UI (task #6), still "
        "pending. Build a LandmarkSet directly via LandmarkSet.from_dict for "
        "non-interactive use."
    )
