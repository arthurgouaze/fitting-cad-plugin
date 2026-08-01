"""End-to-end CLI pipeline tests, run headless against the synthetic mesh.

Exercises cli.main(["fit", ...]) over the full
load -> normalize -> landmarks -> generate -> write path using a
canonical-frame landmark sidecar (no interactive picker, no GUI), and checks the
handled error paths exit non-zero with clean messages rather than tracebacks.

The GUI-backed paths (fit falling through to the interactive picker, and the
view subcommand) are covered with the picker / viewer entry points patched
out, so the wiring is tested without ever opening a window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fitting_cad import cli, io
from fitting_cad.cli import main
from fitting_cad.landmarks import Landmark, LandmarkSet, snap_to_surface

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SYNTHETIC = EXAMPLES / "synthetic_limb.stl"


def _write_landmark_sidecar(path: Path) -> Path:
    """Write a canonical-frame landmark sidecar for the synthetic mesh.

    Landmarks are placed on the NORMALIZED mesh (the frame the pipeline uses):
    anterior is +X (theta=0), posterior/popliteal is -X (theta=pi), ~1/4 down
    the limb, snapped to the surface -- the same construction as the trimline
    tests.
    """
    mesh = io.normalize_orientation(io.load_mesh(SYNTHETIC))
    z = np.asarray(mesh.vertices)[:, 2]
    band_z = float(z.min() + 0.25 * (z.max() - z.min()))

    def at(angle: float) -> list[float]:
        point = snap_to_surface(mesh, [80.0 * np.cos(angle), 80.0 * np.sin(angle), band_z])
        return point.tolist()

    sidecar = {
        Landmark.PATELLAR_TENDON.value: at(0.0),
        Landmark.MEDIAL_CONDYLE.value: at(np.pi / 2 * 0.8),
        Landmark.LATERAL_CONDYLE.value: at(-np.pi / 2 * 0.8),
        Landmark.FIBULAR_HEAD.value: at(np.pi / 2),
        Landmark.POPLITEAL.value: at(np.pi),
    }
    path.write_text(json.dumps(sidecar, indent=2))
    return path


def test_fit_pipeline_writes_closed_loop_json(tmp_path):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    out = tmp_path / "trimline.json"

    code = main(["fit", str(SYNTHETIC), "--landmarks", str(sidecar), "-o", str(out)])

    assert code == 0
    assert out.exists()

    curve = np.asarray(json.loads(out.read_text()), dtype=np.float64)
    # Ordered loop of (N, 3) points, default resolution.
    assert curve.ndim == 2 and curve.shape[1] == 3
    assert curve.shape[0] == 180
    # Closed: end wraps back near the start (gap ~ one segment, not a big jump).
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    closure_gap = np.linalg.norm(curve[-1] - curve[0])
    assert closure_gap < 2.0 * seg.mean()
    # Wraps the axis exactly once.
    angles = np.unwrap(np.arctan2(curve[:, 1], curve[:, 0]))
    assert abs(np.ptp(angles) - 2.0 * np.pi) < 0.2


def test_fit_pipeline_csv_output(tmp_path):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    out = tmp_path / "trimline.csv"

    code = main(["fit", str(SYNTHETIC), "--landmarks", str(sidecar), "-o", str(out)])

    assert code == 0
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "x,y,z"
    assert len(lines) == 1 + 180  # header + points


def test_fit_default_output_beside_scan(tmp_path):
    # Copy the scan into tmp so the default sidecar lands in tmp, not the repo.
    scan = tmp_path / "limb.stl"
    scan.write_bytes(SYNTHETIC.read_bytes())
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")

    code = main(["fit", str(scan), "--landmarks", str(sidecar)])

    assert code == 0
    assert (tmp_path / "limb.trimline.json").exists()


def _landmark_set_for(mesh) -> LandmarkSet:
    """The same canonical-frame landmarks as the sidecar, as a LandmarkSet."""
    z = np.asarray(mesh.vertices)[:, 2]
    band_z = float(z.min() + 0.25 * (z.max() - z.min()))

    def at(angle: float):
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


def test_fit_without_landmarks_runs_the_picker_and_saves_a_sidecar(tmp_path, monkeypatch, capsys):
    """--landmarks omitted falls through to the picker; the picked set persists."""
    scan = tmp_path / "limb.stl"
    scan.write_bytes(SYNTHETIC.read_bytes())
    seen = {}

    def fake_picker(mesh, save_path=None, **kwargs):
        seen["save_path"] = save_path
        landmarks = _landmark_set_for(mesh)
        from fitting_cad.landmarks import save_landmarks_json

        save_landmarks_json(landmarks, save_path)
        return landmarks

    monkeypatch.setattr(cli, "place_landmarks", fake_picker)

    code = main(["fit", str(scan)])

    assert code == 0
    # Default sidecar lands beside the scan, in the canonical frame.
    assert seen["save_path"] == tmp_path / "limb.landmarks.json"
    assert (tmp_path / "limb.landmarks.json").exists()
    assert (tmp_path / "limb.trimline.json").exists()
    assert "Traceback" not in capsys.readouterr().out


def test_fit_save_landmarks_overrides_the_default_path(tmp_path, monkeypatch):
    scan = tmp_path / "limb.stl"
    scan.write_bytes(SYNTHETIC.read_bytes())
    target = tmp_path / "picked" / "session.json"

    def fake_picker(mesh, save_path=None, **kwargs):
        from fitting_cad.landmarks import save_landmarks_json

        landmarks = _landmark_set_for(mesh)
        save_landmarks_json(landmarks, save_path)
        return landmarks

    monkeypatch.setattr(cli, "place_landmarks", fake_picker)

    code = main(["fit", str(scan), "--save-landmarks", str(target)])

    assert code == 0
    assert target.exists()


def test_fit_without_viz_extra_exits_with_actionable_message(monkeypatch, capsys):
    """No pyvista -> a clean message pointing at the extra and --landmarks."""

    def no_pyvista(mesh, save_path=None, **kwargs):
        raise ImportError("pyvista is not installed. Install it with: pip install -e '.[dev,viz]'")

    monkeypatch.setattr(cli, "place_landmarks", no_pyvista)

    code = main(["fit", str(SYNTHETIC)])

    assert code != 0
    out = capsys.readouterr().out
    assert "--landmarks" in out
    assert "viz" in out
    assert "Traceback" not in out


def test_fit_step_output_writes_valid_step(tmp_path, capsys):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    out = tmp_path / "trimline.step"

    code = main(["fit", str(SYNTHETIC), "--landmarks", str(sidecar), "-o", str(out)])

    assert code == 0
    assert out.exists()
    text = out.read_text()
    assert text.startswith("ISO-10303-21;")
    assert text.rstrip().endswith("END-ISO-10303-21;")
    assert "POLYLINE(" in text
    # One CARTESIAN_POINT per trim-loop vertex, plus the appended closing vertex.
    assert text.count("CARTESIAN_POINT") == 180 + 1
    captured = capsys.readouterr().out
    assert "Traceback" not in captured


class _ShowSpy:
    """Stand-in for viewer.show that records what the CLI handed it."""

    def __init__(self):
        self.calls = []

    def __call__(self, mesh, landmarks=None, trimline=None, **kwargs):
        self.calls.append({"mesh": mesh, "landmarks": landmarks, "trimline": trimline, **kwargs})


def test_view_opens_the_normalized_mesh(monkeypatch):
    spy = _ShowSpy()
    monkeypatch.setattr(cli.viewer, "show", spy)

    code = main(["view", str(SYNTHETIC)])

    assert code == 0
    (call,) = spy.calls
    assert call["landmarks"] is None and call["trimline"] is None
    assert call["off_screen"] is False
    # Normalized frame: the proximal rim sits at Z = 0.
    assert np.asarray(call["mesh"].vertices)[:, 2].min() == pytest.approx(0.0, abs=10.0)


def test_view_overlays_landmarks_and_trimline(tmp_path, monkeypatch, capsys):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    spy = _ShowSpy()
    monkeypatch.setattr(cli.viewer, "show", spy)

    code = main(["view", str(SYNTHETIC), "--landmarks", str(sidecar), "--trimline"])

    assert code == 0
    (call,) = spy.calls
    assert call["landmarks"].missing() == []
    assert call["trimline"].shape == (180, 3)
    out = capsys.readouterr().out
    assert "landmark" in out and "trim line" in out


def test_view_screenshot_renders_off_screen(tmp_path, monkeypatch):
    shot = tmp_path / "scene.png"
    spy = _ShowSpy()
    monkeypatch.setattr(cli.viewer, "show", spy)

    code = main(["view", str(SYNTHETIC), "--screenshot", str(shot)])

    assert code == 0
    (call,) = spy.calls
    assert call["off_screen"] is True
    assert call["screenshot"] == shot


def test_view_trimline_without_landmarks_errors(capsys):
    code = main(["view", str(SYNTHETIC), "--trimline"])

    assert code != 0
    out = capsys.readouterr().out
    assert "--landmarks" in out
    assert "Traceback" not in out


def test_view_without_viz_extra_exits_cleanly(monkeypatch, capsys):
    def no_pyvista(*args, **kwargs):
        raise ImportError("pyvista is not installed. Install it with: pip install -e '.[dev,viz]'")

    monkeypatch.setattr(cli.viewer, "show", no_pyvista)

    code = main(["view", str(SYNTHETIC)])

    assert code != 0
    out = capsys.readouterr().out
    assert "viz" in out
    assert "Traceback" not in out
