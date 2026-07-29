"""Mesh I/O and orientation normalization.

Roadmap: Week 5 (mesh pipeline) and Week 7 (export).

Import a limb scan (STL/PLY), normalize its orientation into a canonical frame,
and export results as STL (3D print) or STEP (CAD/CAM). Develop against synthetic
or open limb-scan meshes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import trimesh

from fitting_cad import step_writer

if TYPE_CHECKING:
    pass


# Suffixes trimesh can round-trip as a single surface mesh for this pipeline.
_SUPPORTED_MESH_SUFFIXES = {".stl", ".ply"}

# Suffixes the trim curve can be exported to as CAD/CAM geometry.
_SUPPORTED_CURVE_SUFFIXES = {".step", ".stp"}


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a limb-scan mesh from STL or PLY.

    The scan must be a single triangular surface mesh. Point clouds, empty
    geometry, and multi-body scenes are rejected with a clear error rather than
    silently coerced -- reconstruction and scene splitting are separate steps.

    Args:
        path: Path to an ``.stl`` or ``.ply`` file.

    Returns:
        A single :class:`trimesh.Trimesh` with at least one face.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The suffix is unsupported, or the loaded geometry is not a
            single non-empty surface mesh (e.g. a point cloud or ``Scene``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_MESH_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_MESH_SUFFIXES))
        raise ValueError(
            f"Unsupported mesh format {suffix!r} for {path.name}; "
            f"expected one of: {supported}"
        )

    # process=False keeps the scan verbatim (no vertex merging) so later
    # landmark indices stay reproducible; normalization handles cleanup.
    loaded = trimesh.load(path, force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        # force="mesh" usually concatenates a scene, but an empty scene can slip
        # through. Treat a leftover Scene as multi-body / unsupported input.
        raise ValueError(
            f"{path.name} loaded as a multi-body scene; this pipeline expects a "
            "single surface mesh. Split or reconstruct it first."
        )

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(
            f"{path.name} did not load as a surface mesh (got "
            f"{type(loaded).__name__}); point clouds are not supported."
        )

    if loaded.faces is None or len(loaded.faces) == 0:
        raise ValueError(
            f"{path.name} has no faces; it looks like a point cloud, not a "
            "surface mesh. Reconstruct a surface before importing."
        )

    return loaded


def normalize_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy of the mesh re-oriented into the canonical limb frame.

    Canonical frame:

    * The limb long axis is aligned to **+Z**.
    * The **proximal rim** (the wide, open socket-opening end of a transtibial
      residual limb) sits at **Z = 0**, and the distal end runs toward **+Z**.
    * The origin is the centroid of the proximal-rim cross-section, so trim-line
      rules can be expressed as frame-independent offsets down the limb.

    The long axis is estimated by PCA on the vertices (the principal axis of
    greatest variance). The proximal/distal sign is fixed deterministically: a
    residual limb tapers distally, so the wider end is proximal. We compare the
    cross-sectional spread of vertices in the slab near each end of the axis and
    orient so the **wider end is proximal (Z = 0)** and the narrower end is +Z.

    The input mesh is not mutated; a transformed copy is returned.

    Args:
        mesh: A surface mesh in arbitrary scanner coordinates.

    Returns:
        A new :class:`trimesh.Trimesh` in the canonical limb frame.

    Raises:
        ValueError: The mesh has no vertices.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        raise ValueError("Cannot normalize a mesh with no vertices.")

    # --- 1. Long axis via PCA -------------------------------------------------
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    # Columns of eigvecs are principal axes; largest eigenvalue == long axis.
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, int(np.argmax(eigvals))]
    long_axis = long_axis / np.linalg.norm(long_axis)

    # --- 2. Provisional rotation that maps long_axis -> +Z --------------------
    z_hat = np.array([0.0, 0.0, 1.0])
    rotation = _rotation_between(long_axis, z_hat)

    rotated = centered @ rotation.T  # row vectors: v' = R v

    # --- 3. Deterministic proximal/distal sign --------------------------------
    # Slab near each Z extreme; the wider slab is the proximal rim.
    z = rotated[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    span = z_max - z_min
    slab = max(span * 0.1, 1e-9)  # 10% of length at each end

    low_mask = z <= (z_min + slab)
    high_mask = z >= (z_max - slab)
    low_spread = _xy_spread(rotated[low_mask])
    high_spread = _xy_spread(rotated[high_mask])

    # If the wider end is at high Z, flip so it becomes the low (proximal) end.
    if high_spread > low_spread:
        flip = np.diag([1.0, -1.0, -1.0])  # 180 deg about X keeps +Z handedness
        rotated = rotated @ flip.T
        z = rotated[:, 2]
        z_min = float(z.min())

    # --- 4. Translate proximal-rim centroid to the origin ---------------------
    # Recompute the proximal slab after any flip and use its XY/Z centroid so the
    # rim cross-section centroid lands at the origin.
    z_min = float(rotated[:, 2].min())
    span = float(rotated[:, 2].max()) - z_min
    slab = max(span * 0.1, 1e-9)
    rim_mask = rotated[:, 2] <= (z_min + slab)
    rim_centroid = rotated[rim_mask].mean(axis=0)

    canonical = rotated - rim_centroid

    out = mesh.copy()
    out.vertices = canonical
    return out


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Export a mesh to STL or PLY. Format inferred from the path suffix.

    Args:
        mesh: The mesh to write.
        path: Destination path; suffix selects the format (``.stl`` / ``.ply``).

    Returns:
        The :class:`~pathlib.Path` that was written.

    Raises:
        ValueError: The suffix is not a supported export format.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_MESH_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_MESH_SUFFIXES))
        raise ValueError(
            f"Unsupported export format {suffix!r} for {path.name}; "
            f"expected one of: {supported}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    return path


def export_curve(curve: np.ndarray, path: str | Path) -> Path:
    """Export the ordered closed trim curve as CAD/CAM-ready STEP.

    The loop is written as a single ``POLYLINE`` inside a millimetre AP214
    wireframe shape (see :mod:`fitting_cad.step_writer`). trimesh has no STEP
    writer and a full OCCT bridge (cascadio/OCP/build123d) is far too heavy just
    to serialize a curve, so this uses a small dependency-free text writer.

    Args:
        curve: An ``(N, 3)`` array of ordered loop vertices (mm) in the canonical
            frame. ``N`` must be at least 3.
        path: Destination ``.step`` / ``.stp`` path.

    Returns:
        The :class:`~pathlib.Path` written.

    Raises:
        ValueError: The suffix is not ``.step`` / ``.stp``, or the curve is not a
            valid ``(N >= 3, 3)`` array of finite points.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_CURVE_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_CURVE_SUFFIXES))
        raise ValueError(
            f"Unsupported curve export format {suffix!r} for {path.name}; "
            f"expected one of: {supported}"
        )
    return step_writer.write_step_polyline(curve, path)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a rotation matrix mapping unit vector ``a`` onto unit vector ``b``.

    Uses Rodrigues' formula. Handles the parallel and anti-parallel degenerate
    cases explicitly.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        # Parallel (c ~ +1) -> identity; anti-parallel (c ~ -1) -> 180 deg flip.
        if c > 0:
            return np.eye(3)
        # Pick any axis orthogonal to a for the 180 deg rotation.
        orth = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            orth = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, orth)
        axis = axis / np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)

    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for ``angle`` radians about a unit ``axis``."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    cc = 1.0 - c
    return np.array(
        [
            [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
            [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
            [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
        ]
    )


def _xy_spread(points: np.ndarray) -> float:
    """Mean radial spread of points in the XY plane about their XY centroid.

    A proxy for cross-sectional width used to tell the wide (proximal) end of a
    tapered limb from the narrow (distal) end.
    """
    if points.shape[0] == 0:
        return 0.0
    xy = points[:, :2]
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    return float(radii.mean())
