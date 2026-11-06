"""End-to-end CLI pipeline tests, run headless against the synthetic mesh.

Exercises cli.main(["fit", ...]) over the full
load -> clean -> normalize -> landmarks -> generate -> write path using a
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

from _synthetic import SYNTHETIC, normalized_mesh, synthetic_landmarks
from fitting_cad import cli, io
from fitting_cad.cli import main
from fitting_cad.landmarks import save_landmarks_json
from fitting_cad.trimline import TrimRules


def _write_landmark_sidecar(path: Path) -> Path:
    """Write the synthetic landmarks as a canonical-frame sidecar.

    The pipeline cleans the scan before normalizing it, so the sidecar is placed
    on the uncleaned normalized mesh and relies on snapping to re-seat it.
    """
    return save_landmarks_json(synthetic_landmarks(normalized_mesh()), path)


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


def test_fit_without_landmarks_runs_the_picker(tmp_path, monkeypatch, capsys):
    """--landmarks omitted falls through to the picker; the picked set persists."""
    scan = tmp_path / "limb.stl"
    scan.write_bytes(SYNTHETIC.read_bytes())
    seen = {}

    def fake_picker(mesh, save_path=None, **kwargs):
        seen["save_path"] = save_path
        landmarks = synthetic_landmarks(mesh)
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
        landmarks = synthetic_landmarks(mesh)
        save_landmarks_json(landmarks, save_path)
        return landmarks

    monkeypatch.setattr(cli, "place_landmarks", fake_picker)

    code = main(["fit", str(scan), "--save-landmarks", str(target)])

    assert code == 0
    assert target.exists()


def test_fit_without_viz_extra_exits_cleanly(monkeypatch, capsys):
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


def _spy_on_generate(monkeypatch) -> list:
    """Wrap cli.generate_trimline and record the TrimRules each call received."""
    seen = []
    real = cli.generate_trimline

    def spy(mesh, landmarks, rules=None, **kwargs):
        seen.append(rules)
        return real(mesh, landmarks, rules=rules, **kwargs)

    monkeypatch.setattr(cli, "generate_trimline", spy)
    return seen


def test_fit_rules_default_to_the_dataclass(tmp_path, monkeypatch):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    seen = _spy_on_generate(monkeypatch)

    code = main(["fit", str(SYNTHETIC), "--landmarks", str(sidecar), "-o", str(tmp_path / "t.json")])

    assert code == 0
    assert seen == [TrimRules()]


def test_fit_rules_flags_reach_the_rule_engine(tmp_path, monkeypatch):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    seen = _spy_on_generate(monkeypatch)

    code = main(
        [
            "fit",
            str(SYNTHETIC),
            "--landmarks",
            str(sidecar),
            "-o",
            str(tmp_path / "t.json"),
            "--anterior-offset-mm",
            "22.5",
            "--posterior-height-mm",
            "4",
            "--smoothing",
            "0",
        ]
    )

    assert code == 0
    assert seen == [TrimRules(anterior_offset_mm=22.5, posterior_height_mm=4.0, smoothing=0.0)]


def test_fit_anterior_offset_flag_lowers_anterior(tmp_path):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")

    def anterior_z(offset: str) -> float:
        out = tmp_path / f"trim_{offset}.json"
        args = ["fit", str(SYNTHETIC), "--landmarks", str(sidecar), "-o", str(out)]
        assert main([*args, "--anterior-offset-mm", offset]) == 0
        curve = np.asarray(json.loads(out.read_text()))
        return float(curve[np.argmax(curve[:, 0]), 2])

    assert anterior_z("30") < anterior_z("5")


def test_fit_rejects_non_numeric_rule_values(tmp_path):
    with pytest.raises(SystemExit):
        main(["fit", str(SYNTHETIC), "--anterior-offset-mm", "deep"])


def test_fit_cleans_by_default_and_reports_it(tmp_path, capsys):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")

    code = main(["fit", str(SYNTHETIC), "--landmarks", str(sidecar), "-o", str(tmp_path / "t.json")])

    assert code == 0
    assert "Cleanup: merged" in capsys.readouterr().out


def test_fit_no_clean_skips_cleanup(tmp_path, capsys):
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    args = ["fit", str(SYNTHETIC), "--landmarks", str(sidecar), "-o", str(tmp_path / "t.json")]

    code = main([*args, "--no-clean"])

    assert code == 0
    assert "Cleanup:" not in capsys.readouterr().out


def test_fit_runs_on_a_metre_scaled_scan(tmp_path, capsys):
    metres = io.load_mesh(SYNTHETIC)
    metres.apply_scale(0.001)
    scan = io.export_mesh(metres, tmp_path / "limb_m.obj")
    sidecar = _write_landmark_sidecar(tmp_path / "landmarks.json")
    out = tmp_path / "t.json"

    code = main(["fit", str(scan), "--landmarks", str(sidecar), "-o", str(out)])

    assert code == 0
    assert "metres to mm" in capsys.readouterr().out
    curve = np.asarray(json.loads(out.read_text()))
    # Back in millimetres: the loop is limb-sized, not a fraction of a unit.
    assert np.ptp(curve[:, 0]) > 50.0


def test_fit_max_faces_without_clean_errors(tmp_path, capsys):
    code = main(["fit", str(SYNTHETIC), "--no-clean", "--max-faces", "1000"])
    assert code == 1
    assert "--max-faces needs --clean" in capsys.readouterr().out


def test_view_no_clean_shows_the_raw_vertex_count(monkeypatch):
    spy = _ShowSpy()
    monkeypatch.setattr(cli.viewer, "show", spy)

    assert main(["view", str(SYNTHETIC), "--no-clean"]) == 0
    assert main(["view", str(SYNTHETIC)]) == 0

    raw, cleaned = (len(call["mesh"].vertices) for call in spy.calls)
    assert raw == len(io.load_mesh(SYNTHETIC).vertices)
    assert cleaned < raw
