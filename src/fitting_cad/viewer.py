"""PyVista-based 3D viewer.

Renders the limb mesh, the placed landmarks, and the generated trim curve in one
scene, and provides the drawing primitives the interactive landmark picker
(fitting_cad.landmarks.place_landmarks) builds its session on.

pyvista is an optional dependency (the viz extra): the headless pipeline
(fitting-cad fit --landmarks ...) never imports it. Every entry point here
therefore imports it lazily via require_pyvista, which raises an
actionable ImportError when the extra is missing.

Coordinate frame: everything is drawn in whatever frame the caller passes. In
practice that is the canonical limb frame (long axis +Z, proximal rim at
Z = 0, anterior +X) produced by
fitting_cad.io.normalize_orientation, so the default camera looks at the
limb from the anterolateral side with +Z up.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import trimesh

    from fitting_cad.landmarks import LandmarkSet

# --- Scene palette ---------------------------------------------------------
# Green limb, blue landmarks, red trim line, on a dark background.
LIMB_COLOR = "#008000"
LANDMARK_COLOR = "#1f6feb"
LANDMARK_LABEL_COLOR = "#eef2f7"
TRIMLINE_COLOR = "#e03131"
BACKGROUND_COLOR = "#1c2027"

# Overlay sizes as a fraction of the mesh bounding-box diagonal, so the scene
# reads the same whether the scan is in millimetres or metres.
LANDMARK_RADIUS_FRAC = 0.015
TRIMLINE_RADIUS_FRAC = 0.006

_VIZ_HINT = (
    "The 3D viewer and the interactive landmark picker need the optional 'viz' "
    "extra. Install it with:  pip install -e '.[dev,viz]'  . The headless "
    "pipeline (fitting-cad fit --landmarks PATH) runs without it."
)


def require_pyvista() -> Any:
    """Import and return the pyvista module, or explain how to install it.

    Returns:
        The imported pyvista module.

    Raises:
        ImportError: pyvista is not installed, with the install command for the
            optional viz extra and the headless alternative.
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(f"pyvista is not installed. {_VIZ_HINT}") from exc
    return pv


def mesh_to_polydata(mesh: trimesh.Trimesh) -> Any:
    """Convert a trimesh.Trimesh to a pyvista.PolyData surface.

    Built explicitly from vertices + a VTK face array ([3, i, j, k, ...])
    rather than going through pv.wrap, so the conversion does not depend on
    pyvista's optional trimesh interoperability.

    Args:
        mesh: A triangular surface mesh.

    Returns:
        A pyvista.PolyData with the same vertices and triangles.

    Raises:
        ValueError: The mesh has no triangular faces (e.g. a point cloud).
    """
    pv = require_pyvista()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] != 3:
        raise ValueError(
            "Cannot render a mesh without triangular faces; reconstruct a surface "
            "from the point cloud first."
        )
    padded = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, padded)


def scene_scale(mesh: trimesh.Trimesh) -> float:
    """Return the mesh bounding-box diagonal, used to size the overlays.

    Args:
        mesh: The mesh being rendered.

    Returns:
        The bounding-box diagonal length, or 1.0 for a degenerate mesh.
    """
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    return diagonal if diagonal > 0.0 else 1.0


def add_limb_surface(plotter: Any, mesh: trimesh.Trimesh, name: str = "limb") -> Any:
    """Add the limb surface to plotter (LIMB_COLOR, edges off).

    Args:
        plotter: A pyvista.Plotter.
        mesh: The limb mesh to render.
        name: Actor name, so re-adding replaces rather than stacks.

    Returns:
        The added pyvista actor.
    """
    return plotter.add_mesh(
        mesh_to_polydata(mesh),
        color=LIMB_COLOR,
        show_edges=False,
        smooth_shading=True,
        specular=0.15,
        specular_power=12,
        name=name,
    )


def add_landmark_glyph(
    plotter: Any,
    name: str,
    point: np.ndarray,
    radius: float,
    label: bool = True,
) -> None:
    """Add one landmark as a sphere glyph with an optional text label.

    Actors are named landmark-<name> / landmark-label-<name> so the
    interactive picker can remove a single landmark on re-pick.

    Args:
        plotter: A pyvista.Plotter.
        name: Landmark name (e.g. "patellar_tendon"), used for the label and
            the actor names.
        point: The (3,) landmark position.
        radius: Sphere radius in model units.
        label: Whether to draw the landmark name beside the glyph.
    """
    pv = require_pyvista()
    centre = np.asarray(point, dtype=np.float64).reshape(3)
    plotter.add_mesh(
        pv.Sphere(radius=radius, center=centre, theta_resolution=24, phi_resolution=24),
        color=LANDMARK_COLOR,
        smooth_shading=True,
        name=f"landmark-{name}",
    )
    if label:
        plotter.add_point_labels(
            centre.reshape(1, 3),
            [name.replace("_", " ")],
            font_size=11,
            text_color=LANDMARK_LABEL_COLOR,
            shape_color=LANDMARK_COLOR,
            shape_opacity=0.55,
            show_points=False,
            always_visible=True,
            reset_camera=False,
            name=f"landmark-label-{name}",
        )


def remove_landmark_glyph(plotter: Any, name: str) -> None:
    """Remove a landmark glyph and its label, if present (used on re-pick).

    Args:
        plotter: A pyvista.Plotter.
        name: The landmark name passed to add_landmark_glyph.
    """
    # remove_actor returns False for an unknown name rather than raising.
    for actor_name in (f"landmark-{name}", f"landmark-label-{name}"):
        plotter.remove_actor(actor_name, render=False)


def add_landmark_glyphs(plotter: Any, landmarks: LandmarkSet, radius: float) -> None:
    """Add every placed landmark in landmarks as a labelled sphere glyph.

    Args:
        plotter: A pyvista.Plotter.
        landmarks: The placed landmark set.
        radius: Sphere radius in model units.
    """
    for landmark, point in landmarks.points.items():
        add_landmark_glyph(plotter, landmark.value, point, radius=radius)


def add_trimline(plotter: Any, curve: np.ndarray, radius: float, name: str = "trimline") -> Any:
    """Add the trim curve as a closed tube in a contrasting colour.

    Args:
        plotter: A pyvista.Plotter.
        curve: An (N, 3) ordered loop of points (as returned by
            fitting_cad.trimline.generate_trimline).
        radius: Tube radius in model units.
        name: Actor name.

    Returns:
        The added pyvista actor.

    Raises:
        ValueError: curve is not an (N, 3) array with at least 2 points.
    """
    pv = require_pyvista()
    points = np.asarray(curve, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
        raise ValueError(f"Trim curve must be an (N, 3) array with N >= 2, got {points.shape}.")
    # Close the loop for rendering if the caller's curve does not repeat its first
    # point (generate_trimline returns an open list of ordered loop points).
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[:1]])
    tube = pv.lines_from_points(points).tube(radius=radius, n_sides=16)
    return plotter.add_mesh(tube, color=TRIMLINE_COLOR, smooth_shading=True, name=name)


def frame_camera(plotter: Any, mesh: trimesh.Trimesh) -> None:
    """Point the camera at the limb from the anterolateral side, +Z up.

    Args:
        plotter: A pyvista.Plotter.
        mesh: The mesh to frame.
    """
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    centre = (bounds[0] + bounds[1]) / 2.0
    radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) / 2.0, 1e-6)
    # Anterior is +X in the canonical frame; offset laterally and slightly above
    # the centroid so the anterior trim relief is visible without hiding the rim.
    position = centre + np.array([2.2 * radius, -1.6 * radius, 0.6 * radius])
    plotter.camera_position = [tuple(position), tuple(centre), (0.0, 0.0, 1.0)]
    plotter.reset_camera()


def show(
    mesh: trimesh.Trimesh,
    landmarks: LandmarkSet | None = None,
    trimline: np.ndarray | None = None,
    *,
    off_screen: bool = False,
    screenshot: str | Path | None = None,
    window_size: tuple[int, int] = (1100, 800),
    title: str = "fitting-cad",
) -> None:
    """Show the limb mesh with optional landmark and trim-curve overlays.

    The limb renders as an opaque surface, landmarks as labelled spheres,
    and the trim curve as a contrasting tube, with an orientation-axes widget and
    a camera framed on the limb with +Z up.

    Pass the normalized mesh (see
    fitting_cad.io.normalize_orientation) together with landmarks and a
    curve from the same frame; this function does no frame conversion.

    Args:
        mesh: The limb mesh to render.
        landmarks: Optional placed landmarks to overlay.
        trimline: Optional (N, 3) ordered trim loop to overlay.
        off_screen: Render without opening a window. Use with screenshot for
            headless capture (and in tests).
        screenshot: Optional PNG path to write. Parent directories are created.
        window_size: Render size in pixels.
        title: Window title.

    Raises:
        ImportError: pyvista is not installed (see the viz extra).
        ValueError: The mesh has no faces, or trimline is not an (N, 3)
            loop.
    """
    pv = require_pyvista()

    shot_path: Path | None = None
    if screenshot is not None:
        shot_path = Path(screenshot)
        shot_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size, title=title)
    try:
        plotter.set_background(BACKGROUND_COLOR)
        add_limb_surface(plotter, mesh)

        scale = scene_scale(mesh)
        if landmarks is not None and landmarks.points:
            add_landmark_glyphs(plotter, landmarks, radius=LANDMARK_RADIUS_FRAC * scale)
        if trimline is not None and len(trimline):
            add_trimline(plotter, trimline, radius=TRIMLINE_RADIUS_FRAC * scale)

        plotter.add_axes()
        frame_camera(plotter, mesh)

        if off_screen:
            if shot_path is not None:
                plotter.screenshot(str(shot_path))
            else:
                plotter.render()
        else:
            plotter.show(screenshot=str(shot_path) if shot_path is not None else False)
    finally:
        plotter.close()
