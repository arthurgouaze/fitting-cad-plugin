"""End-to-end CLI pipeline tests, run headless against the synthetic mesh.

Exercises ``cli.main(["fit", ...])`` over the full
``load -> normalize -> landmarks -> generate -> write`` path using a
canonical-frame landmark sidecar (no interactive picker, no GUI), and checks the
handled error paths (omitted ``--landmarks``, unsupported STEP output) exit
non-zero with clean messages rather than tracebacks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fitting_cad import io
from fitting_cad.cli import main
from fitting_cad.landmarks import Landmark, snap_to_surface

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


def test_fit_without_landmarks_exits_with_actionable_message(tmp_path, capsys):
    code = main(["fit", str(SYNTHETIC)])

    assert code != 0
    out = capsys.readouterr().out
    assert "--landmarks" in out
    assert "not implemented" in out.lower()
    # No raw traceback leaked.
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


def test_view_subcommand_clean_stub_message(capsys):
    code = main(["view", str(SYNTHETIC)])

    assert code != 0
    out = capsys.readouterr().out
    assert "not implemented" in out.lower()
    assert "Traceback" not in out
