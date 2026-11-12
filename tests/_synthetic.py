# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Arthur Gouaze
"""Synthetic-limb helpers shared by the test modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from limbform import io
from limbform.landmarks import Landmark, LandmarkSet, snap_to_surface

SYNTHETIC = Path(__file__).resolve().parent.parent / "examples" / "synthetic_limb.stl"


def normalized_mesh() -> trimesh.Trimesh:
    return io.normalize_orientation(io.load_mesh(SYNTHETIC))


def synthetic_landmarks(mesh: trimesh.Trimesh) -> LandmarkSet:
    """The required landmarks in a band ~1/4 down the limb, snapped on-surface.

    Anterior is +X (theta=0); posterior/popliteal is -X (theta=pi).
    """
    z = np.asarray(mesh.vertices)[:, 2]
    band_z = float(z.min() + 0.25 * (z.max() - z.min()))

    def at(angle: float) -> np.ndarray:
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
