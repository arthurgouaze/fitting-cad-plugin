# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Arthur Gouaze
"""Anatomical landmark placement and the non-interactive plumbing around it.

Roadmap: Week 5 (landmark-placement step).

The clinician places ~6-8 anatomical landmarks on the normalized limb mesh.
These drive the rule-based trim-curve generation in limbform.trimline.

This module keeps the non-interactive parts (the LandmarkSet
container, a nearest-vertex snap helper, the sidecar reader/writer, and the
_PickSession state machine) free of any GUI / GL dependency so they are
testable headless. place_landmarks imports pyvista lazily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import trimesh

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class Landmark(str, Enum):
    """The transtibial landmark set a prosthetist references for the trim line."""

    PATELLA = "patella"
    PATELLAR_TENDON = "patellar_tendon"
    TIBIAL_TUBERCLE = "tibial_tubercle"
    FIBULAR_HEAD = "fibular_head"
    MEDIAL_CONDYLE = "medial_condyle"
    LATERAL_CONDYLE = "lateral_condyle"
    POPLITEAL = "popliteal"  # posterior reference


# Landmarks the first-pass trim-line rule engine needs to anchor its z(theta)
# profile. The patellar tendon anchors the anterior relief, the popliteal anchors
# the posterior (higher) trim, and the fibular head and condyles fix the
# medial/lateral wall heights. The patella and the tibial tubercle are useful
# context for a clinician but optional: the current rule does not need them.
REQUIRED_LANDMARKS: tuple[Landmark, ...] = (
    Landmark.PATELLAR_TENDON,
    Landmark.FIBULAR_HEAD,
    Landmark.MEDIAL_CONDYLE,
    Landmark.LATERAL_CONDYLE,
    Landmark.POPLITEAL,
)

# Order the interactive picker walks the clinician through. Declaration order of
# Landmark, i.e. anterior/proximal first (patella, patellar tendon,
# tibial tubercle), then the medial/lateral walls, then the posterior popliteal
# reference last.
PICK_ORDER: tuple[Landmark, ...] = tuple(Landmark)


@dataclass
class LandmarkSet:
    """Resolved model coordinates for each placed landmark."""

    points: dict[Landmark, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, points: Mapping[Landmark, np.ndarray | list[float] | tuple[float, ...]]
    ) -> LandmarkSet:
        """Build a LandmarkSet from a mapping of a landmark to a point.

        Each value is converted to a (3,) float64 array. This is the
        non-interactive entry point for coordinates obtained elsewhere.

        Args:
            points: Mapping from Landmark to a 3D point.

        Returns:
            A LandmarkSet with copied, validated points.

        Raises:
            ValueError: A value is not a 3-vector.
        """
        resolved: dict[Landmark, np.ndarray] = {}
        for landmark, point in points.items():
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
            if arr.shape != (3,):
                raise ValueError(
                    f"Landmark {landmark.value!r} must be a 3D point, got a shape "
                    f"{np.asarray(point).shape}."
                )
            resolved[landmark] = arr
        return cls(points=resolved)

    def missing(self) -> list[Landmark]:
        """Required landmarks (see REQUIRED_LANDMARKS) not yet placed.

        Returns:
            The required landmarks absent from points, in canonical order.
        """
        return [lm for lm in REQUIRED_LANDMARKS if lm not in self.points]


def snap_to_surface(mesh: trimesh.Trimesh, point: np.ndarray) -> np.ndarray:
    """Snap a 3D point to the nearest vertex of mesh.

    A clinician's pick (or a programmatic landmark) rarely lands exactly on a
    mesh vertex; snapping to the nearest vertex keeps landmark positions
    reproducible and tied to concrete mesh geometry. Uses trimesh's vertex
    KD-tree (mesh.kdtree) to remain GUI-free and fast.

    Args:
        mesh: The mesh whose vertices are candidate snap targets.
        point: A 3D point in the same coordinate frame as mesh.

    Returns:
        A (3,) float64 copy of the nearest mesh vertex.

    Raises:
        ValueError: The point is not a 3-vector, or the mesh has no vertices.
    """
    p = np.asarray(point, dtype=np.float64).reshape(-1)
    if p.shape != (3,):
        raise ValueError(f"point must be a 3D point, got a shape {np.asarray(point).shape}.")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.shape[0] == 0:
        raise ValueError("Cannot snap to a mesh with no vertices.")

    # mesh.kdtree is a cached scipy cKDTree over the vertices.
    _, index = mesh.kdtree.query(p)
    return vertices[int(index)].copy()


def load_landmarks_json(path: str | Path, mesh: trimesh.Trimesh | None = None) -> LandmarkSet:
    """Load a landmark sidecar (name -> [x, y, z]) into a LandmarkSet.

    The sidecar is a JSON object mapping each landmark name to a 3-element list,
    e.g.:

        {
            "patellar_tendon": [12.0, 0.0, 40.0],
            "popliteal": [-30.0, 0.0, 55.0],
            ...
        }

    Coordinates are read in the canonical limb frame returned by
    limbform.io.normalize_orientation (long axis +Z, proximal rim at Z = 0,
    origin at the proximal-rim centroid), not raw scanner coordinates. The
    pipeline normalizes the mesh before generating the trim line, so a sidecar
    in scanner coordinates would not line up with the surface.

    Args:
        path: Path to the JSON sidecar.
        mesh: Optional normalized mesh. When given, each landmark is snapped
            to the nearest mesh vertex via snap_to_surface, so small
            off-surface offsets in the sidecar are tied back to the geometry.

    Returns:
        A LandmarkSet in the canonical frame.

    Raises:
        FileNotFoundError: The sidecar path does not exist.
        ValueError: The JSON is not an object, a key is not a known landmark
            name, or a value is not a 3-vector.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Landmark sidecar not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        # Malformed sidecar content, not a bad argument type, so ValueError
        # keeps the documented contract and the CLI error path intact.
        raise ValueError(  # noqa: TRY004
            f"{path.name} must be a JSON object mapping landmark name -> [x, y, z], "
            f"got {type(raw).__name__}."
        )

    valid_names = {lm.value: lm for lm in Landmark}
    points: dict[Landmark, np.ndarray] = {}
    for name, value in raw.items():
        if name not in valid_names:
            known = ", ".join(sorted(valid_names))
            raise ValueError(f"Unknown landmark {name!r} in {path.name}; known names: {known}.")
        landmark = valid_names[name]
        point = np.asarray(value, dtype=np.float64).reshape(-1)
        if mesh is not None:
            point = snap_to_surface(mesh, point)
        points[landmark] = point

    return LandmarkSet.from_dict(points)


def save_landmarks_json(landmark_set: LandmarkSet, path: str | Path) -> Path:
    """Write a LandmarkSet to a JSON sidecar (name -> [x, y, z]).

    The inverse of load_landmarks_json, and subject to the same
    canonical-frame convention: the points written here are assumed to be in
    the canonical limb frame (placed on the normalized mesh), because that is
    the frame the pipeline generates the trim line in. Keys are written in
    Landmark declaration order so sidecars diff cleanly.

    Args:
        landmark_set: The placed landmarks to persist.
        path: Destination JSON path. Parent directories are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        landmark.value: [float(c) for c in np.asarray(point, dtype=np.float64).reshape(3)]
        for landmark, point in landmark_set.points.items()
    }
    ordered = {lm.value: payload[lm.value] for lm in Landmark if lm.value in payload}
    path.write_text(json.dumps(ordered, indent=2) + "\n")
    return path


class _PickSession:
    """Headless state machine behind the interactive picker.

    Holds everything the picker does that is not rendering: which landmark is
    currently being asked for, snapping an incoming pick to the surface,
    recording it, undoing the last pick, skipping optional landmarks, and
    reporting the on-screen prompt. Keeping this GL-free is what makes the
    picker's logic unit-testable (feeding it synthetic pick events) while the GUI
    itself is used manually.
    """

    def __init__(self, mesh: trimesh.Trimesh, order: Iterable[Landmark] | None = None) -> None:
        """Start a picking session over order (defaults to PICK_ORDER).

        Args:
            mesh: The normalized mesh picks are snapped to.
            order: The landmark sequence to walk through.

        Raises:
            ValueError: order is empty or omits a required landmark.
        """
        self.mesh = mesh
        self.order: tuple[Landmark, ...] = tuple(order) if order is not None else PICK_ORDER
        if not self.order:
            raise ValueError("Pick order must contain at least one landmark.")
        skipped_required = [lm for lm in REQUIRED_LANDMARKS if lm not in self.order]
        if skipped_required:
            names = ", ".join(lm.value for lm in skipped_required)
            raise ValueError(f"Pick order omits required landmarks: {names}.")
        self.points: dict[Landmark, np.ndarray] = {}
        self.index = 0

    @property
    def current(self) -> Landmark | None:
        """The landmark the clinician is being asked to place, or None if done."""
        return self.order[self.index] if self.index < len(self.order) else None

    @property
    def done(self) -> bool:
        """True once every landmark in the pick order has been placed or skipped."""
        return self.index >= len(self.order)

    def record(self, point: np.ndarray) -> tuple[Landmark, np.ndarray]:
        """Snap point to the surface, store it as the current landmark, and advance.

        Args:
            point: The raw 3D pick position (typically a ray-surface hit).

        Returns:
            (landmark, snapped_point).

        Raises:
            ValueError: The session is already finished, or point is not a
                3-vector.
        """
        landmark = self.current
        if landmark is None:
            raise ValueError("All landmarks in the pick order have been placed.")
        snapped = snap_to_surface(self.mesh, point)
        self.points[landmark] = snapped
        self.index += 1
        return landmark, snapped

    def undo(self) -> Landmark | None:
        """Step back one slot and drop its landmark so it can be re-picked.

        Returns:
            The landmark that was stepped back to (now unplaced), or None if
            the session is still at the first slot.
        """
        if self.index == 0:
            return None
        self.index -= 1
        landmark = self.order[self.index]
        self.points.pop(landmark, None)
        return landmark

    def skip(self) -> Landmark:
        """Advance past the current landmark without placing it.

        Only optional landmarks (those outside REQUIRED_LANDMARKS) may be
        skipped, because the rule engine cannot anchor its height profile
        without the required set.

        Returns:
            The landmark that was skipped.

        Raises:
            ValueError: The session is finished, or the current landmark is
                required.
        """
        landmark = self.current
        if landmark is None:
            raise ValueError("All landmarks in the pick order have been placed.")
        if landmark in REQUIRED_LANDMARKS:
            raise ValueError(
                f"{landmark.value} is required by the trim-line rule engine and cannot be skipped."
            )
        self.index += 1
        return landmark

    def missing(self) -> list[Landmark]:
        """Required landmarks not yet placed in this session."""
        return [lm for lm in REQUIRED_LANDMARKS if lm not in self.points]

    def result(self) -> LandmarkSet:
        """Return the placed landmarks as a LandmarkSet."""
        return LandmarkSet.from_dict(self.points)

    def prompt(self, note: str | None = None) -> str:
        """The on-screen guidance text for the current state.

        Args:
            note: Optional extra line (e.g. why a skip was refused).

        Returns:
            A multi-line prompt naming the current target landmark and the keys.
        """
        keys = "left click = place   |   u = undo last   |   n = skip optional   |   q = finish"
        landmark = self.current
        if landmark is None:
            head = f"All {len(self.points)} landmark(s) have been placed, finishing."
        else:
            optional = "" if landmark in REQUIRED_LANDMARKS else "  (optional)"
            head = (
                f"[{self.index + 1}/{len(self.order)}] Click the "
                f"{landmark.value.replace('_', ' ')}{optional}"
            )
        lines = [head, keys]
        if note:
            lines.append(note)
        return "\n".join(lines)


def place_landmarks(
    mesh: trimesh.Trimesh,
    save_path: str | Path | None = None,
    *,
    order: Iterable[Landmark] | None = None,
) -> LandmarkSet:
    """Open the interactive picker and return the placed landmark set.

    Opens a pyvista window showing the limb surface and walks the clinician
    through PICK_ORDER one landmark at a time while naming the current target
    on screen:

    - left click on the surface places the current landmark. The pick is
      snapped to the nearest mesh vertex (snap_to_surface), drawn as a
      labelled sphere, and the prompt advances to the next landmark.
    - "u" undoes the last pick (steps back one slot so it can be re-picked).
    - "n" skips the current landmark, allowed only for landmarks outside
      REQUIRED_LANDMARKS.
    - "q" (or closing the window) finishes early; the window also closes
      automatically once the whole pick order is exhausted.

    Run this on the NORMALIZED mesh (the output of
    limbform.io.normalize_orientation), never on the raw scan. The
    returned points, and the sidecar written when save_path is given, are
    therefore in the canonical limb frame, which is the frame
    load_landmarks_json and the trim-line rule engine expect. Picking on
    a raw scan produces a sidecar that will not line up with the pipeline.

    The interactive path is exercised manually (it needs a GL window); the
    pick -> snap -> record logic behind it lives in _PickSession and is
    unit-tested headless.

    Args:
        mesh: The normalized limb mesh to pick on.
        save_path: Optional JSON sidecar path. When given, the placed set is
            persisted via save_landmarks_json for reproducibility.
        order: Optional custom landmark sequence; defaults to PICK_ORDER.

    Returns:
        The placed LandmarkSet, in the canonical frame.

    Raises:
        ImportError: pyvista is not installed (see the viz extra).
        ValueError: The session was finished with required landmarks unplaced.
    """
    from limbform import viewer

    pv = viewer.require_pyvista()

    session = _PickSession(mesh, order=order)
    radius = viewer.LANDMARK_RADIUS_FRAC * viewer.scene_scale(mesh)

    plotter = pv.Plotter(window_size=(1100, 800), title="limbform - place landmarks")
    plotter.set_background(viewer.BACKGROUND_COLOR)
    viewer.add_limb_surface(plotter, mesh)
    plotter.add_axes()
    viewer.frame_camera(plotter, mesh)

    def refresh(note: str | None = None) -> None:
        plotter.add_text(
            session.prompt(note),
            position="upper_left",
            font_size=11,
            color=viewer.LANDMARK_LABEL_COLOR,
            name="picker-prompt",
        )

    def on_pick(point: Any, *_: Any) -> None:
        if session.done:
            return
        landmark, snapped = session.record(np.asarray(point, dtype=np.float64).reshape(3))
        viewer.add_landmark_glyph(plotter, landmark.value, snapped, radius=radius)
        refresh()
        if session.done:
            plotter.close()

    def on_undo() -> None:
        landmark = session.undo()
        if landmark is None:
            refresh("Nothing to undo.")
            return
        viewer.remove_landmark_glyph(plotter, landmark.value)
        refresh()

    def on_skip() -> None:
        try:
            landmark = session.skip()
        except ValueError as exc:
            refresh(str(exc))
            return
        refresh(f"Skipped {landmark.value.replace('_', ' ')}.")
        if session.done:
            plotter.close()

    plotter.add_key_event("u", on_undo)
    plotter.add_key_event("n", on_skip)
    plotter.enable_surface_point_picking(
        callback=on_pick,
        show_message=False,
        show_point=False,
        left_clicking=True,
    )
    refresh()
    plotter.show()

    missing = session.missing()
    if missing:
        names = ", ".join(lm.value for lm in missing)
        raise ValueError(
            f"Landmark picking finished with required landmarks unplaced: {names}. "
            "Re-run the picker and place them, or supply a sidecar with --landmarks."
        )

    landmark_set = session.result()
    if save_path is not None:
        save_landmarks_json(landmark_set, save_path)
    return landmark_set
