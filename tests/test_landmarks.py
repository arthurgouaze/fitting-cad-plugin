"""Tests for the non-interactive landmark plumbing.

The interactive picker (:func:`landmarks.place_landmarks`) is a GUI task and is
not exercised here; these tests cover the headless container, the surface-snap
helper, and the required-landmark report against the synthetic mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitting_cad import io
from fitting_cad.landmarks import (
    REQUIRED_LANDMARKS,
    Landmark,
    LandmarkSet,
    place_landmarks,
    snap_to_surface,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SYNTHETIC = EXAMPLES / "synthetic_limb.stl"


def _normalized_mesh():
    return io.normalize_orientation(io.load_mesh(SYNTHETIC))


def test_from_dict_coerces_points():
    lm = LandmarkSet.from_dict(
        {
            Landmark.PATELLAR_TENDON: [1, 2, 3],
            Landmark.POPLITEAL: (4.0, 5.0, 6.0),
        }
    )
    assert lm.points[Landmark.PATELLAR_TENDON].dtype == np.float64
    assert np.allclose(lm.points[Landmark.PATELLAR_TENDON], [1, 2, 3])
    assert np.allclose(lm.points[Landmark.POPLITEAL], [4, 5, 6])


def test_from_dict_rejects_non_3d_point():
    with pytest.raises(ValueError, match="3D point"):
        LandmarkSet.from_dict({Landmark.POPLITEAL: [1.0, 2.0]})


def test_missing_reports_required_subset():
    empty = LandmarkSet()
    assert empty.missing() == list(REQUIRED_LANDMARKS)

    partial = LandmarkSet.from_dict({Landmark.POPLITEAL: [0, 0, 0]})
    missing = partial.missing()
    assert Landmark.POPLITEAL not in missing
    assert Landmark.PATELLAR_TENDON in missing


def test_missing_empty_when_all_required_present():
    lm = LandmarkSet.from_dict({k: [0, 0, 0] for k in REQUIRED_LANDMARKS})
    assert lm.missing() == []


def test_snap_to_surface_returns_a_mesh_vertex():
    mesh = _normalized_mesh()
    vertices = np.asarray(mesh.vertices)
    # A point offset from a known vertex should snap back onto a real vertex.
    target = vertices[100]
    snapped = snap_to_surface(mesh, target + np.array([0.3, -0.2, 0.1]))
    assert snapped.shape == (3,)
    # The snapped point must be exactly one of the mesh vertices.
    assert np.min(np.linalg.norm(vertices - snapped, axis=1)) == 0.0


def test_snap_to_surface_picks_nearest():
    mesh = _normalized_mesh()
    vertices = np.asarray(mesh.vertices)
    target = vertices[250]
    snapped = snap_to_surface(mesh, target)
    assert np.allclose(snapped, target)


def test_snap_rejects_bad_point():
    mesh = _normalized_mesh()
    with pytest.raises(ValueError, match="3D point"):
        snap_to_surface(mesh, [1.0, 2.0])


def test_place_landmarks_is_stubbed():
    mesh = _normalized_mesh()
    with pytest.raises(NotImplementedError):
        place_landmarks(mesh)
