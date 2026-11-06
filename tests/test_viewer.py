"""Viewer tests, headless only.

The viewer needs the optional viz extra plus a working off-screen GL stack,
neither of which CI has. Everything here is skipped unless pyvista imports and
a probe subprocess proves off-screen rendering works on this host. The probe
runs out-of-process because a broken GL stack aborts the interpreter rather than
raising, which would otherwise take pytest down with it.

The interactive window itself is never opened here; it is exercised manually.
"""

from __future__ import annotations

import functools
import subprocess
import sys

import numpy as np
import pytest

from _synthetic import normalized_mesh, synthetic_landmarks
from fitting_cad.trimline import generate_trimline

pytest.importorskip("pyvista")

from fitting_cad import viewer

_GL_PROBE = (
    "import pyvista as pv;"
    "p = pv.Plotter(off_screen=True, window_size=(64, 64));"
    "p.add_mesh(pv.Sphere());"
    "p.render();"
    "p.close()"
)


@functools.cache
def _offscreen_gl_available() -> bool:
    """Whether this host can render off-screen (probed in a throwaway process)."""
    try:
        probe = subprocess.run(
            [sys.executable, "-c", _GL_PROBE],
            capture_output=True,
            timeout=180,
            check=False,  # a non-zero exit is the "no GL" answer, not an error
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _require_gl() -> None:
    if not _offscreen_gl_available():
        pytest.skip("off-screen GL rendering is unavailable on this host")


# --- Data-level helpers (no rendering, no GL) ------------------------------


def test_mesh_to_polydata_preserves_topology():
    mesh = normalized_mesh()
    poly = viewer.mesh_to_polydata(mesh)
    assert poly.n_points == len(mesh.vertices)
    assert poly.n_cells == len(mesh.faces)


def test_scene_scale_is_the_bounding_box_diagonal():
    mesh = normalized_mesh()
    bounds = np.asarray(mesh.bounds)
    assert viewer.scene_scale(mesh) == pytest.approx(float(np.linalg.norm(bounds[1] - bounds[0])))


# --- Off-screen render (needs GL) ------------------------------------------


def test_show_offscreen_writes_screenshot(tmp_path):
    _require_gl()
    mesh = normalized_mesh()
    landmarks = synthetic_landmarks(mesh)
    curve = generate_trimline(mesh, landmarks)
    shot = tmp_path / "scene.png"

    viewer.show(
        mesh,
        landmarks=landmarks,
        trimline=curve,
        off_screen=True,
        screenshot=shot,
        window_size=(480, 360),
    )

    assert shot.exists()
    # A non-trivial render, not a blank/0-byte file.
    assert shot.stat().st_size > 2000
    assert shot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_show_offscreen_bare_mesh(tmp_path):
    _require_gl()
    shot = tmp_path / "bare.png"
    viewer.show(normalized_mesh(), off_screen=True, screenshot=shot, window_size=(320, 240))
    assert shot.exists() and shot.stat().st_size > 1000


def test_add_trimline_rejects_bad_curve():
    _require_gl()
    pv = viewer.require_pyvista()
    plotter = pv.Plotter(off_screen=True, window_size=(64, 64))
    try:
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            viewer.add_trimline(plotter, np.zeros((4, 2)), radius=1.0)
    finally:
        plotter.close()
