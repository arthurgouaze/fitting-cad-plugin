"""PyVista-based 3D viewer.

Roadmap: Week 5 (viewer) and Week 8 (clinician wrapper).

Render the limb mesh, placed landmarks, and the generated trim curve. Hosts the
interactive picking used by :func:`fitting_cad.landmarks.place_landmarks`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import trimesh

    from fitting_cad.landmarks import LandmarkSet


def show(
    mesh: "trimesh.Trimesh",
    landmarks: "LandmarkSet | None" = None,
    trimline: "np.ndarray | None" = None,
) -> None:
    """Open a pyvista window showing the mesh and any provided overlays.

    TODO(week5): wrap trimesh -> pyvista (pv.wrap); add landmark glyphs with
    labels and the trim curve as a tube; sensible default camera down +Z.
    """
    raise NotImplementedError
