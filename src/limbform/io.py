"""Mesh I/O and orientation normalization.

Roadmap: Week 5 (mesh pipeline) and Week 7 (export).

Import a limb scan (STL/PLY/OBJ), clean it, normalize its orientation into a
canonical frame, and export results as STL (3D print) or STEP (CAD/CAM).
Develop against synthetic or open limb-scan meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from limbform import step_writer

# Suffixes trimesh can round-trip as a single surface mesh for this pipeline.
_SUPPORTED_MESH_SUFFIXES = {".stl", ".ply", ".obj"}

# Suffixes the trim curve can be exported to as CAD/CAM geometry.
_SUPPORTED_CURVE_SUFFIXES = {".step", ".stp"}

# A residual limb is roughly 150-500 mm long. A bounding box whose longest side
# is below this many units can only be a scan exported in metres.
_METRE_SCAN_MAX_EXTENT = 0.5

# Boundary loops up to this many edges are scan dropouts and get capped. Longer
# loops are left open: on a real scan the longest one is usually the deliberate
# proximal opening, which must not be closed off.
_DEFAULT_MAX_HOLE_EDGES = 64


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a limb-scan mesh from STL, PLY, or OBJ.

    The scan must be a single triangular surface mesh. Point clouds, empty
    geometry, and multi-body scenes are rejected with a clear error rather than
    silently coerced. Reconstruction and scene splitting are separate steps.

    Args:
        path: Path to an .stl, .ply, or .obj file.

    Returns:
        A single trimesh.Trimesh with at least one face.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The suffix is unsupported, or the loaded geometry is not a
            single non-empty surface mesh (e.g. a point cloud or Scene).
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
        # force="mesh" concatenates most scenes, but an empty one slips through.
        # ValueError rather than TypeError here and below: the path is valid and
        # the file content is unusable, and the CLI error handler keys on
        # ValueError.
        raise ValueError(  # noqa: TRY004
            f"{path.name} loaded as a multi-body scene; this pipeline expects a "
            "single surface mesh. Split or reconstruct it first."
        )

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(  # noqa: TRY004
            f"{path.name} did not load as a surface mesh (got "
            f"{type(loaded).__name__}); point clouds are not supported."
        )

    if len(loaded.faces) == 0:
        raise ValueError(
            f"{path.name} has no faces; it looks like a point cloud, not a "
            "surface mesh. Reconstruct a surface before importing."
        )

    return loaded


@dataclass
class CleanReport:
    """What clean_mesh changed, so the caller can log or display it.

    Attributes:
        merged_vertices: Vertices removed by merging coincident positions.
        degenerate_faces: Zero-area faces dropped.
        duplicate_faces: Faces dropped because another face used the same
            three vertices.
        unreferenced_vertices: Vertices dropped because no face used them.
        winding_fixed: True when face winding was inconsistent and got
            repaired so neighbouring faces agree.
        holes_filled: Boundary loops capped with a fan of new faces.
        holes_left_open: Boundary loops left open, either longer than the
            hole-size limit (such as the proximal opening) or not a simple loop.
        scale_factor: Factor applied to the coordinates; 1000.0 when the
            scan was detected as metres, otherwise 1.0.
        faces_before_decimation: Face count before decimation, or None when
            no decimation ran.
        faces_after: Face count of the cleaned mesh.
    """

    merged_vertices: int = 0
    degenerate_faces: int = 0
    duplicate_faces: int = 0
    unreferenced_vertices: int = 0
    winding_fixed: bool = False
    holes_filled: int = 0
    holes_left_open: int = 0
    scale_factor: float = 1.0
    faces_before_decimation: int | None = None
    faces_after: int = 0

    def summary(self) -> str:
        """One-line, human-readable account of the cleanup."""
        parts = [
            f"merged {self.merged_vertices} vertices",
            (
                f"dropped {self.degenerate_faces} degenerate and "
                f"{self.duplicate_faces} duplicate faces"
            ),
            f"dropped {self.unreferenced_vertices} unreferenced vertices",
        ]
        if self.winding_fixed:
            parts.append("fixed face winding")
        parts.append(f"filled {self.holes_filled} holes ({self.holes_left_open} left open)")
        if self.scale_factor != 1.0:
            parts.append(f"scaled x{self.scale_factor:g} (metres to mm)")
        if self.faces_before_decimation is not None:
            parts.append(
                f"decimated {self.faces_before_decimation} -> {self.faces_after} faces"
            )
        return "; ".join(parts)


def clean_mesh(
    mesh: trimesh.Trimesh,
    *,
    max_hole_edges: int = _DEFAULT_MAX_HOLE_EDGES,
    max_faces: int | None = None,
) -> tuple[trimesh.Trimesh, CleanReport]:
    """Return a repaired copy of a raw scan and a report of what changed.

    Steps, in order: merge coincident vertices; drop zero-area and duplicate
    faces; drop vertices no face uses; make face winding consistent; cap every
    boundary loop of at most max_hole_edges edges with a fan around its
    centroid; rescale scans exported in metres to millimetres; and, when
    max_faces is given, decimate to that face budget.

    Long boundary loops stay open on purpose. A transtibial scan is usually
    open at the proximal end and that opening is part of the geometry, not a
    defect.

    The metre check is a bounding-box heuristic: a longest side under 0.5 units
    cannot be a limb in millimetres, so the scan is scaled by 1000. Scans in
    centimetres are not detected.

    The input mesh is not mutated.

    Args:
        mesh: The scan as loaded by load_mesh.
        max_hole_edges: Largest boundary loop, in edges, that is treated as a
            hole and capped. Must be at least 3.
        max_faces: Optional face budget. When the cleaned mesh has more
            faces, it is decimated with quadric edge collapse, which needs the
            optional fast-simplification package (the decimate extra).

    Returns:
        A (cleaned_mesh, report) tuple.

    Raises:
        ValueError: max_hole_edges or max_faces is out of range, or nothing
            is left after dropping degenerate faces.
        ImportError: Decimation was requested but fast-simplification is not
            installed.
    """
    if max_hole_edges < 3:
        raise ValueError(f"max_hole_edges must be >= 3, got {max_hole_edges}.")
    if max_faces is not None and max_faces < 4:
        raise ValueError(f"max_faces must be >= 4, got {max_faces}.")

    report = CleanReport()
    out = mesh.copy()

    before = len(out.vertices)
    out.merge_vertices()
    report.merged_vertices = before - len(out.vertices)

    nondegenerate = out.nondegenerate_faces()
    report.degenerate_faces = int((~nondegenerate).sum())
    out.update_faces(nondegenerate)

    unique = out.unique_faces()
    report.duplicate_faces = int((~unique).sum())
    out.update_faces(unique)

    if len(out.faces) == 0:
        raise ValueError("No faces left after dropping degenerate faces.")

    before = len(out.vertices)
    out.remove_unreferenced_vertices()
    report.unreferenced_vertices = before - len(out.vertices)

    if not out.is_winding_consistent:
        trimesh.repair.fix_winding(out)
        report.winding_fixed = True

    report.holes_filled, report.holes_left_open = _fill_small_holes(out, max_hole_edges)

    if float(out.extents.max()) < _METRE_SCAN_MAX_EXTENT:
        report.scale_factor = 1000.0
        out.apply_scale(report.scale_factor)

    if max_faces is not None and len(out.faces) > max_faces:
        try:
            import fast_simplification  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Decimation needs the optional fast-simplification package; "
                "install it with: pip install -e '.[decimate]'"
            ) from exc
        report.faces_before_decimation = len(out.faces)
        out = out.simplify_quadric_decimation(face_count=max_faces)

    report.faces_after = len(out.faces)
    return out, report


def normalize_orientation(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy of the mesh re-oriented into the canonical limb frame.

    Canonical frame:

    - The limb long axis is aligned to +Z.
    - The proximal rim (the wide, open socket-opening end of a transtibial
      residual limb) sits at Z = 0, and the distal end runs toward +Z.
    - The origin is the centroid of the proximal-rim cross-section, so trim-line
      rules can be expressed as frame-independent offsets down the limb.

    The long axis is estimated by PCA on the vertices (the principal axis of
    greatest variance). The proximal/distal sign is fixed deterministically: a
    residual limb tapers distally, so the wider end is proximal. The
    cross-sectional spread of vertices in the slab near each end of the axis is
    compared, and the mesh is oriented so the wider end is proximal (Z = 0)
    and the narrower end is +Z.

    The input mesh is not mutated; a transformed copy is returned.

    Args:
        mesh: A surface mesh in arbitrary scanner coordinates.

    Returns:
        A new trimesh.Trimesh in the canonical limb frame.

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

    # --- 4. Translate proximal-rim centroid to the origin ---------------------
    # A flip leaves the span unchanged, so only the low end needs recomputing.
    rim_mask = rotated[:, 2] <= float(rotated[:, 2].min()) + slab
    rim_centroid = rotated[rim_mask].mean(axis=0)

    canonical = rotated - rim_centroid

    out = mesh.copy()
    out.vertices = canonical
    return out


def export_mesh(mesh: trimesh.Trimesh, path: str | Path) -> Path:
    """Export a mesh to STL, PLY, or OBJ. Format inferred from the path suffix.

    Args:
        mesh: The mesh to write.
        path: Destination path; suffix selects the format (.stl / .ply / .obj).

    Returns:
        The pathlib.Path that was written.

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

    The loop is written as a single POLYLINE inside a millimetre AP214
    wireframe shape (see limbform.step_writer). trimesh has no STEP
    writer and a full OCCT bridge (cascadio/OCP/build123d) is far too heavy just
    to serialize a curve, so this uses a small dependency-free text writer.

    Args:
        curve: An (N, 3) array of ordered loop vertices (mm) in the canonical
            frame. N must be at least 3.
        path: Destination .step / .stp path.

    Returns:
        The pathlib.Path written.

    Raises:
        ValueError: The suffix is not .step / .stp, or the curve is not a
            valid (N >= 3, 3) array of finite points.
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


def _fill_small_holes(mesh: trimesh.Trimesh, max_edges: int) -> tuple[int, int]:
    """Cap boundary loops of at most max_edges edges in place.

    Each loop is closed with a fan of triangles around a new vertex at the loop
    centroid, wound against the boundary edge so the cap faces the same way as
    the surrounding surface. Loops through a non-manifold boundary vertex are
    left open.

    Returns:
        A (filled, left_open) count of boundary loops.
    """
    boundary = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)
    if len(boundary) == 0:
        return 0, 0

    # mesh.edges keeps each face's winding, so (a, b) runs along the boundary
    # in the direction the adjacent face traverses it.
    directed = mesh.edges[boundary]
    successor: dict[int, int] = {}
    branching: set[int] = set()
    for a, b in directed.tolist():
        if a in successor:
            branching.add(a)
        successor[a] = b

    loops: list[list[int]] = []
    visited: set[int] = set()
    for start, current in successor.items():
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        while current != start and current in successor and current not in visited:
            loop.append(current)
            visited.add(current)
            current = successor[current]
        loops.append(loop if current == start else [])

    vertices = [np.asarray(mesh.vertices)]
    faces = [np.asarray(mesh.faces)]
    next_index = len(mesh.vertices)
    filled = 0
    for loop in loops:
        if not loop or len(loop) > max_edges or branching.intersection(loop):
            continue
        ring = np.asarray(loop)
        vertices.append(np.asarray(mesh.vertices)[ring].mean(axis=0, keepdims=True))
        faces.append(np.column_stack([np.roll(ring, -1), ring, np.full(len(ring), next_index)]))
        next_index += 1
        filled += 1

    if filled:
        mesh.vertices = np.vstack(vertices)
        mesh.faces = np.vstack(faces)
    return filled, len(loops) - filled


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a rotation matrix mapping unit vector a onto unit vector b.

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
        return _axis_angle(np.cross(a, orth), np.pi)

    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for angle radians about a unit axis."""
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
    xy = points[:, :2]
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    return float(radii.mean())
