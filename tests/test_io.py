"""I/O-layer tests against the synthetic limb mesh.

These exercise load -> normalize -> export round-trip on
examples/synthetic_limb.stl (no patient data). Full-pipeline tests (load ->
landmarks -> trimline -> export) arrive once landmark/trimline land.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from fitting_cad import io

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SYNTHETIC = EXAMPLES / "synthetic_limb.stl"


def _principal_axis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    return eigvecs[:, int(np.argmax(eigvals))]


def _xy_spread(points: np.ndarray) -> float:
    xy = points[:, :2]
    return float(np.linalg.norm(xy - xy.mean(axis=0), axis=1).mean())


def test_synthetic_mesh_present():
    assert SYNTHETIC.exists(), (
        f"{SYNTHETIC} missing; regenerate with examples/make_synthetic_limb.py"
    )


def test_load_mesh_returns_trimesh():
    mesh = io.load_mesh(SYNTHETIC)
    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.faces) > 0
    assert len(mesh.vertices) > 0


def test_load_mesh_missing_file():
    with pytest.raises(FileNotFoundError):
        io.load_mesh(EXAMPLES / "does_not_exist.stl")


def test_load_mesh_unsupported_suffix(tmp_path):
    bogus = tmp_path / "scan.xyz"
    bogus.write_text("not a mesh")
    with pytest.raises(ValueError, match="Unsupported mesh format"):
        io.load_mesh(bogus)


def test_load_mesh_rejects_point_cloud(tmp_path):
    # A PLY with vertices but no faces is a point cloud -> must be rejected.
    cloud = trimesh.points.PointCloud(np.random.default_rng(0).random((50, 3)))
    path = tmp_path / "cloud.ply"
    cloud.export(path)
    with pytest.raises(ValueError):
        io.load_mesh(path)


def test_normalize_aligns_long_axis_to_z():
    mesh = io.load_mesh(SYNTHETIC)
    normalized = io.normalize_orientation(mesh)
    axis = _principal_axis(np.asarray(normalized.vertices))
    # Principal axis should be ~ +/-Z (sign handled by the proximal/distal rule).
    assert abs(abs(axis[2]) - 1.0) < 1e-3
    assert abs(axis[0]) < 1e-2
    assert abs(axis[1]) < 1e-2


def test_normalize_puts_proximal_rim_at_origin():
    mesh = io.load_mesh(SYNTHETIC)
    normalized = io.normalize_orientation(mesh)
    v = np.asarray(normalized.vertices)
    z = v[:, 2]
    slab = (z.max() - z.min()) * 0.1
    rim = v[z <= z.min() + slab]
    # Proximal-rim cross-section centroid sits at the origin.
    assert np.linalg.norm(rim.mean(axis=0)) < 1.0  # within 1 mm


def test_normalize_proximal_end_is_wider():
    # Convention: the wider (proximal) end is at low Z; the limb tapers to +Z.
    mesh = io.load_mesh(SYNTHETIC)
    normalized = io.normalize_orientation(mesh)
    v = np.asarray(normalized.vertices)
    z = v[:, 2]
    slab = (z.max() - z.min()) * 0.1
    rim = v[z <= z.min() + slab]
    distal = v[z >= z.max() - slab]
    assert _xy_spread(rim) > _xy_spread(distal)


def test_normalize_does_not_mutate_input():
    mesh = io.load_mesh(SYNTHETIC)
    before = np.asarray(mesh.vertices).copy()
    io.normalize_orientation(mesh)
    assert np.allclose(np.asarray(mesh.vertices), before)


@pytest.mark.parametrize("suffix", [".stl", ".ply"])
def test_export_roundtrip_preserves_counts(tmp_path, suffix):
    mesh = io.load_mesh(SYNTHETIC)
    out = tmp_path / f"roundtrip{suffix}"
    written = io.export_mesh(mesh, out)
    assert written == out
    assert out.exists()
    reloaded = io.load_mesh(out)
    assert len(reloaded.vertices) == len(mesh.vertices)
    assert len(reloaded.faces) == len(mesh.faces)


def test_export_unsupported_suffix(tmp_path):
    mesh = io.load_mesh(SYNTHETIC)
    with pytest.raises(ValueError, match="Unsupported export format"):
        io.export_mesh(mesh, tmp_path / "out.obj")


def _make_circle(n: int = 24, radius: float = 40.0, z: float = 12.5) -> np.ndarray:
    """An ordered, unclosed (N, 3) loop: a horizontal circle at height z."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.full(n, z)])


def _parse_step_points(text: str) -> np.ndarray:
    """Extract CARTESIAN_POINT coordinates from STEP text, in file order."""
    coords = re.findall(
        r"CARTESIAN_POINT\('',\(([^)]*)\)\)", text
    )
    return np.array([[float(v) for v in c.split(",")] for c in coords], dtype=np.float64)


@pytest.mark.parametrize("suffix", [".step", ".stp"])
def test_export_curve_writes_valid_step(tmp_path, suffix):
    curve = _make_circle()
    out = tmp_path / f"trimline{suffix}"

    written = io.export_curve(curve, out)

    assert written == out
    assert out.exists()
    text = out.read_text()
    # Non-trivial file with the ISO-10303-21 envelope and expected entities.
    assert len(text) > 500
    assert text.startswith("ISO-10303-21;")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    assert "FILE_SCHEMA(('AUTOMOTIVE_DESIGN" in text
    assert "SI_UNIT(.MILLI.,.METRE.)" in text  # millimetre units
    assert "POLYLINE(" in text


def test_export_curve_closes_the_loop(tmp_path):
    curve = _make_circle(n=24)
    out = io.export_curve(curve, tmp_path / "loop.step")
    pts = _parse_step_points(out.read_text())
    # First vertex repeated as the last so the wire closes.
    assert len(pts) == 25
    assert np.allclose(pts[0], pts[-1])


def test_export_curve_coordinates_match_input(tmp_path):
    # Spot-check that the numbers written to the file are the curve's own points.
    curve = _make_circle(n=12, radius=37.5, z=8.25)
    out = io.export_curve(curve, tmp_path / "loop.step")
    pts = _parse_step_points(out.read_text())
    assert np.allclose(pts[:-1], curve)  # drop the appended closing vertex


def test_export_curve_rejects_bad_suffix(tmp_path):
    with pytest.raises(ValueError, match="Unsupported curve export format"):
        io.export_curve(_make_circle(), tmp_path / "loop.iges")


def test_export_curve_rejects_too_few_points(tmp_path):
    with pytest.raises(ValueError, match="at least 3 points"):
        io.export_curve(np.zeros((2, 3)), tmp_path / "loop.step")


def test_export_curve_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        io.export_curve(np.zeros((10, 2)), tmp_path / "loop.step")
