"""Tests for the trim-line rule engine against the synthetic mesh.

Builds a ``LandmarkSet`` programmatically on the normalized synthetic mesh (no
GUI), runs :func:`generate_trimline`, and asserts the loop is closed, on the
surface, correctly sized, and monotonic around the axis -- plus that a rule
parameter (posterior trim height) actually moves the curve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitting_cad import io
from fitting_cad.landmarks import Landmark, LandmarkSet, snap_to_surface
from fitting_cad.trimline import TrimRules, generate_trimline, validate_trimline

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SYNTHETIC = EXAMPLES / "synthetic_limb.stl"
RESOLUTION = 180


def _normalized_mesh():
    return io.normalize_orientation(io.load_mesh(SYNTHETIC))


def _synthetic_landmarks(mesh) -> LandmarkSet:
    """Place the required landmarks in a band ~1/4 down the limb, snapped on-surface.

    Anterior is +X (theta=0); posterior/popliteal is -X (theta=pi).
    """
    z = np.asarray(mesh.vertices)[:, 2]
    band_z = float(z.min() + 0.25 * (z.max() - z.min()))

    def at(angle: float):
        # Cast outward from the axis and snap to the nearest surface vertex.
        return snap_to_surface(mesh, [80.0 * np.cos(angle), 80.0 * np.sin(angle), band_z])

    return LandmarkSet.from_dict(
        {
            Landmark.PATELLAR_TENDON: at(0.0),
            Landmark.MEDIAL_CONDYLE: at(np.pi / 2 * 0.8),
            Landmark.LATERAL_CONDYLE: at(-np.pi / 2 * 0.8),
            Landmark.FIBULAR_HEAD: at(np.pi / 2),
            Landmark.POPLITEAL: at(np.pi),
        }
    )


def test_landmarks_complete():
    mesh = _normalized_mesh()
    lm = _synthetic_landmarks(mesh)
    assert lm.missing() == []


def test_generate_returns_requested_resolution():
    mesh = _normalized_mesh()
    loop = generate_trimline(mesh, _synthetic_landmarks(mesh), resolution=RESOLUTION)
    assert loop.shape == (RESOLUTION, 3)


def test_generate_loop_is_closed():
    mesh = _normalized_mesh()
    loop = generate_trimline(mesh, _synthetic_landmarks(mesh), resolution=RESOLUTION)
    # Closure gap should be ~ the uniform segment spacing, not a large jump.
    seg = np.linalg.norm(np.diff(loop, axis=0), axis=1)
    closure_gap = np.linalg.norm(loop[-1] - loop[0])
    assert closure_gap < 2.0 * seg.mean()


def test_generate_points_on_surface():
    mesh = _normalized_mesh()
    loop = generate_trimline(mesh, _synthetic_landmarks(mesh), resolution=RESOLUTION)
    import trimesh

    _, distance, _ = trimesh.proximity.closest_point(mesh, loop)
    # Every point should sit within a small distance of the mesh surface.
    assert float(np.max(distance)) < 1.0  # mm


def test_generate_angles_monotonic_around_axis():
    mesh = _normalized_mesh()
    loop = generate_trimline(mesh, _synthetic_landmarks(mesh), resolution=RESOLUTION)
    angles = np.arctan2(loop[:, 1], loop[:, 0])
    unwrapped = np.unwrap(angles)
    # Strictly increasing (or decreasing) -> wraps the axis exactly once.
    diffs = np.diff(unwrapped)
    assert np.all(diffs > 0)
    assert abs(np.ptp(unwrapped) - 2.0 * np.pi) < 0.2


def test_posterior_height_rule_takes_effect():
    mesh = _normalized_mesh()
    lm = _synthetic_landmarks(mesh)
    theta = np.linspace(0.0, 2.0 * np.pi, RESOLUTION, endpoint=False)
    post = int(np.argmin(np.abs(theta - np.pi)))  # posterior = -X

    low = generate_trimline(
        mesh, lm, TrimRules(posterior_height_mm=5.0, smoothing=0.0), resolution=RESOLUTION
    )
    high = generate_trimline(
        mesh, lm, TrimRules(posterior_height_mm=40.0, smoothing=0.0), resolution=RESOLUTION
    )
    # A higher posterior_height_mm raises the posterior trim (toward +Z).
    assert high[post, 2] > low[post, 2] + 10.0


def test_anterior_offset_rule_lowers_anterior():
    mesh = _normalized_mesh()
    lm = _synthetic_landmarks(mesh)
    theta = np.linspace(0.0, 2.0 * np.pi, RESOLUTION, endpoint=False)
    ant = int(np.argmin(np.abs(theta - 0.0)))  # anterior = +X

    shallow = generate_trimline(
        mesh, lm, TrimRules(anterior_offset_mm=0.0, smoothing=0.0), resolution=RESOLUTION
    )
    deep = generate_trimline(
        mesh, lm, TrimRules(anterior_offset_mm=30.0, smoothing=0.0), resolution=RESOLUTION
    )
    # A larger anterior offset lowers the anterior trim (toward the rim, -Z).
    assert deep[ant, 2] < shallow[ant, 2] - 10.0


def test_generate_requires_landmarks():
    mesh = _normalized_mesh()
    with pytest.raises(ValueError, match="missing landmarks"):
        generate_trimline(mesh, LandmarkSet(), resolution=RESOLUTION)


def test_generate_rejects_tiny_resolution():
    mesh = _normalized_mesh()
    with pytest.raises(ValueError, match="resolution"):
        generate_trimline(mesh, _synthetic_landmarks(mesh), resolution=4)


def test_validate_metrics_are_sane():
    mesh = _normalized_mesh()
    loop = generate_trimline(mesh, _synthetic_landmarks(mesh), resolution=RESOLUTION)
    metrics = validate_trimline(loop, mesh)

    assert metrics["point_count"] == RESOLUTION
    assert metrics["max_distance_to_surface"] < 1.0
    assert metrics["self_intersections"] == 0
    # Closure gap ~ uniform spacing for an evenly resampled loop.
    assert metrics["closure_gap"] < 2.0 * metrics["mean_segment_length"]


def test_validate_rejects_bad_curve():
    mesh = _normalized_mesh()
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        validate_trimline(np.zeros((2, 3)), mesh)
