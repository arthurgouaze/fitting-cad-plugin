"""Minimal, dependency-free STEP (ISO 10303-21) writer for a 3D trim curve.

Roadmap: Week 7 (export). Emits the ordered closed trim loop as CAD/CAM-ready
STEP so it can be opened in FreeCAD / Fusion / a CAM package and used as a
reference wire for socket construction.

Why hand-rolled instead of an OCCT bridge:

- cascadio (trimesh's OCCT bridge) only reads STEP -> GLB; it cannot write
  STEP, and a wireframe curve would not survive a mesh conversion anyway.
- build123d / OCP pull in the multi-hundred-megabyte cadquery-ocp
  OCCT bindings whose wheels are fragile on Python 3.13 / NixOS -- far too heavy
  just to serialize a polyline.

STEP is a plain-text format, so a small, well-formed AP214 writer is the lightest
thing that produces a valid curve entity, with no dependency and no native code.

Entity schema (AP214 AUTOMOTIVE_DESIGN, a widely-imported STEP flavor):

- A minimal product structure (APPLICATION_CONTEXT -> PRODUCT ->
  PRODUCT_DEFINITION -> PRODUCT_DEFINITION_SHAPE) so the curve opens as a
  named part rather than orphan geometry.
- A units context in millimetres (SI_UNIT(.MILLI.,.METRE.)) -- the limb
  frame is metric mm, and CAM tools need explicit units.
- One CARTESIAN_POINT per loop vertex.
- A single POLYLINE through those points. A polyline preserves the exact
  generated vertices (no spline re-approximation); the trim loop is already
  densely sampled, so a polyline is the faithful representation.
- The polyline is carried by a GEOMETRIC_CURVE_SET inside a
  GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION, linked to the product
  via SHAPE_DEFINITION_REPRESENTATION.

The loop is closed explicitly: the first vertex is repeated as the last polyline
point so the wire forms a closed rim.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Loops shorter than this are not a meaningful closed curve to hand to CAD/CAM.
_MIN_LOOP_POINTS = 3

# Tolerance (mm) for deciding the loop is already closed (first ~= last vertex).
_CLOSE_TOL = 1e-6


def write_step_polyline(
    curve: np.ndarray,
    path: str | Path,
    *,
    name: str = "trimline",
) -> Path:
    """Write an ordered closed 3D loop to a STEP file as a single polyline.

    Args:
        curve: An (N, 3) array of ordered loop vertices, in millimetres, in
            the canonical limb frame. N must be at least 3.
        path: Destination .step / .stp path. Parent dirs are created.
        name: Product / curve label embedded in the file.

    Returns:
        The pathlib.Path written.

    Raises:
        ValueError: curve is not an (N, 3) array with N >= 3, or
            contains non-finite coordinates.
    """
    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"Trim curve must be an (N, 3) array of points; got shape {pts.shape}."
        )
    if pts.shape[0] < _MIN_LOOP_POINTS:
        raise ValueError(
            f"Trim curve needs at least {_MIN_LOOP_POINTS} points to form a closed "
            f"loop; got {pts.shape[0]}."
        )
    if not np.isfinite(pts).all():
        raise ValueError("Trim curve contains non-finite coordinates (nan/inf).")

    # Close the loop: repeat the first vertex unless the loop already closes.
    if np.linalg.norm(pts[-1] - pts[0]) > _CLOSE_TOL:
        pts = np.vstack([pts, pts[0]])

    text = _build_step_text(pts, name=name)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _build_step_text(pts: np.ndarray, *, name: str) -> str:
    """Assemble the full ISO-10303-21 document for the closed polyline pts."""
    lines: list[str] = []

    def emit(entity_id: int, body: str) -> None:
        lines.append(f"#{entity_id}={body};")

    # --- Product structure ---------------------------------------------------
    emit(1, "APPLICATION_CONTEXT('automotive design')")
    emit(
        2,
        "APPLICATION_PROTOCOL_DEFINITION('international standard',"
        "'automotive_design',2010,#1)",
    )
    emit(3, "PRODUCT_CONTEXT('',#1,'mechanical')")
    emit(4, f"PRODUCT('{name}','{name}','trim line',(#3))")
    emit(5, "PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design')")
    emit(6, "PRODUCT_DEFINITION_FORMATION('','',#4)")
    emit(7, "PRODUCT_DEFINITION('design','',#6,#5)")
    emit(8, "PRODUCT_DEFINITION_SHAPE('','',#7)")

    # --- Units + geometric context (millimetres) -----------------------------
    emit(9, "( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
    emit(10, "( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    emit(11, "( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    emit(
        12,
        "UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#9,"
        "'distance_accuracy_value','edge curve length')",
    )
    emit(
        13,
        "( GEOMETRIC_REPRESENTATION_CONTEXT(3) "
        "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#12)) "
        "GLOBAL_UNIT_ASSIGNED_CONTEXT((#9,#10,#11)) REPRESENTATION_CONTEXT('','') )",
    )

    # --- Geometry: one CARTESIAN_POINT per vertex, then a POLYLINE ------------
    first_point_id = 14
    point_ids: list[int] = []
    for i, (x, y, z) in enumerate(pts):
        pid = first_point_id + i
        point_ids.append(pid)
        emit(pid, f"CARTESIAN_POINT('',({_fmt(x)},{_fmt(y)},{_fmt(z)}))")

    polyline_id = first_point_id + len(pts)
    refs = ",".join(f"#{pid}" for pid in point_ids)
    emit(polyline_id, f"POLYLINE('{name}',({refs}))")

    curve_set_id = polyline_id + 1
    emit(curve_set_id, f"GEOMETRIC_CURVE_SET('{name}',(#{polyline_id}))")

    rep_id = curve_set_id + 1
    emit(
        rep_id,
        "GEOMETRICALLY_BOUNDED_WIREFRAME_SHAPE_REPRESENTATION("
        f"'{name}',(#{curve_set_id}),#13)",
    )

    sdr_id = rep_id + 1
    emit(sdr_id, f"SHAPE_DEFINITION_REPRESENTATION(#8,#{rep_id})")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    header = (
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION(('transtibial socket trim line'),'2;1');\n"
        f"FILE_NAME('{name}.step','{timestamp}',(''),(''),"
        "'fitting-cad','fitting-cad','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\n"
        "ENDSEC;\n"
    )
    body = "DATA;\n" + "\n".join(lines) + "\nENDSEC;\n"
    return header + body + "END-ISO-10303-21;\n"


def _fmt(value: float) -> str:
    """Format a float as a valid STEP real literal (must contain a decimal point).

    Python's repr can emit exponent forms like 1e-06 whose mantissa has no
    decimal point, which is invalid STEP. This normalizes those to 1.E-06 and
    appends a trailing "." to bare integers, preserving full repr precision.
    """
    s = repr(float(value))
    if "e" in s or "E" in s:
        mantissa, exponent = re.split("[eE]", s)
        if "." not in mantissa:
            mantissa += "."
        return f"{mantissa}E{exponent}"
    if "." not in s:
        s += "."
    return s
