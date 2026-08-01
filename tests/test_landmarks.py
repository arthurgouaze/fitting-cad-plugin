"""Tests for the non-interactive landmark plumbing.

Covers the headless container, the surface-snap helper, the required-landmark
report, the JSON sidecar round-trip, and the picker's pick -> snap -> record
state machine (landmarks._PickSession) fed synthetic pick events.

The interactive window driven by landmarks.place_landmarks is a GUI path
and is exercised manually, not here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fitting_cad import io
from fitting_cad.landmarks import (
    PICK_ORDER,
    REQUIRED_LANDMARKS,
    Landmark,
    LandmarkSet,
    _PickSession,
    load_landmarks_json,
    save_landmarks_json,
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


# --- JSON sidecar round-trip -----------------------------------------------


def test_save_load_landmarks_json_round_trip(tmp_path):
    mesh = _normalized_mesh()
    vertices = np.asarray(mesh.vertices)
    original = LandmarkSet.from_dict(
        {lm: vertices[i * 37] for i, lm in enumerate(REQUIRED_LANDMARKS)}
    )

    path = save_landmarks_json(original, tmp_path / "picked.landmarks.json")
    assert path.exists()

    reloaded = load_landmarks_json(path, mesh=mesh)
    assert set(reloaded.points) == set(original.points)
    for landmark, point in original.points.items():
        assert np.allclose(reloaded.points[landmark], point)


def test_save_landmarks_json_writes_declaration_order(tmp_path):
    import json

    scrambled = LandmarkSet.from_dict(
        {
            Landmark.POPLITEAL: [0, 0, 0],
            Landmark.PATELLA: [1, 1, 1],
            Landmark.FIBULAR_HEAD: [2, 2, 2],
        }
    )
    path = save_landmarks_json(scrambled, tmp_path / "nested" / "lm.json")
    keys = list(json.loads(path.read_text()))
    assert keys == [Landmark.PATELLA.value, Landmark.FIBULAR_HEAD.value, Landmark.POPLITEAL.value]


# --- Picker state machine (no GUI) -----------------------------------------


def _pick_near(mesh, index: int) -> np.ndarray:
    """A synthetic pick event: slightly off a known vertex, as a click would be."""
    return np.asarray(mesh.vertices)[index] + np.array([0.2, -0.15, 0.1])


def test_pick_session_walks_the_order_and_snaps():
    mesh = _normalized_mesh()
    vertices = np.asarray(mesh.vertices)
    session = _PickSession(mesh)

    assert session.current == PICK_ORDER[0]
    landmark, snapped = session.record(_pick_near(mesh, 100))

    assert landmark == PICK_ORDER[0]
    assert session.current == PICK_ORDER[1]
    # Recorded position is an actual mesh vertex, not the raw click.
    assert np.min(np.linalg.norm(vertices - snapped, axis=1)) == 0.0
    assert np.allclose(session.points[landmark], snapped)


def test_pick_session_completes_and_reports_result():
    mesh = _normalized_mesh()
    session = _PickSession(mesh)
    for i, _ in enumerate(PICK_ORDER):
        session.record(_pick_near(mesh, 50 + 20 * i))

    assert session.done
    assert session.current is None
    assert session.missing() == []
    result = session.result()
    assert set(result.points) == set(PICK_ORDER)
    assert result.missing() == []


def test_pick_session_undo_allows_repick():
    mesh = _normalized_mesh()
    session = _PickSession(mesh)
    session.record(_pick_near(mesh, 10))
    _, second = session.record(_pick_near(mesh, 20))

    undone = session.undo()

    assert undone == PICK_ORDER[1]
    assert PICK_ORDER[1] not in session.points
    assert session.current == PICK_ORDER[1]

    _, repicked = session.record(_pick_near(mesh, 300))
    assert not np.allclose(repicked, second)
    assert session.current == PICK_ORDER[2]


def test_pick_session_undo_at_start_is_a_no_op():
    session = _PickSession(_normalized_mesh())
    assert session.undo() is None
    assert session.index == 0


def test_pick_session_record_after_completion_raises():
    mesh = _normalized_mesh()
    session = _PickSession(mesh)
    for i, _ in enumerate(PICK_ORDER):
        session.record(_pick_near(mesh, 5 * i + 1))
    with pytest.raises(ValueError, match="have been placed"):
        session.record(_pick_near(mesh, 400))


def test_pick_session_skips_optional_but_not_required():
    mesh = _normalized_mesh()
    # PATELLA is optional, PATELLAR_TENDON is required.
    session = _PickSession(mesh, order=(Landmark.PATELLA, *REQUIRED_LANDMARKS))

    assert session.skip() == Landmark.PATELLA
    assert Landmark.PATELLA not in session.points
    assert session.current == REQUIRED_LANDMARKS[0]

    with pytest.raises(ValueError, match="cannot be skipped"):
        session.skip()


def test_pick_session_missing_tracks_required_only():
    mesh = _normalized_mesh()
    session = _PickSession(mesh)
    assert session.missing() == list(REQUIRED_LANDMARKS)
    # PICK_ORDER starts with the optional PATELLA; placing it leaves all
    # required landmarks outstanding.
    session.record(_pick_near(mesh, 12))
    assert session.missing() == list(REQUIRED_LANDMARKS)


def test_pick_session_rejects_order_missing_required():
    with pytest.raises(ValueError, match="omits required landmarks"):
        _PickSession(_normalized_mesh(), order=(Landmark.PATELLA,))


def test_pick_session_prompt_names_the_current_landmark():
    mesh = _normalized_mesh()
    session = _PickSession(mesh)

    prompt = session.prompt()
    assert PICK_ORDER[0].value.replace("_", " ") in prompt
    assert "left click" in prompt
    assert "u = undo" in prompt

    assert "why not" in session.prompt("why not")
