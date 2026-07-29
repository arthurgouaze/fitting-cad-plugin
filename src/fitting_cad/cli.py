"""Command-line entry point.

Roadmap: Week 8 (clinician wrapper).

Thin wrapper that wires the pipeline together::

    load -> normalize -> place landmarks -> generate trim line -> export

The ``fit`` subcommand runs headless: pass a landmark sidecar with
``--landmarks`` so no interactive picker is required (the picker, #6, is still a
stub). The ``view`` subcommand needs the viewer (#7) and is still a stub.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fitting_cad import io
from fitting_cad.landmarks import load_landmarks_json, place_landmarks
from fitting_cad.trimline import generate_trimline, validate_trimline

# Curve output formats. Ordered-point formats are written here; .step/.stp route
# to io.export_curve, which writes a CAD/CAM-ready STEP polyline.
_POINT_SUFFIXES = {".json", ".csv", ".npy"}
_STEP_SUFFIXES = {".step", ".stp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fitting-cad",
        description="Automated transtibial socket trim-line generation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser(
        "fit",
        help="run the full trim-line pipeline on a scan",
        description=(
            "Load a limb scan, normalize it into the canonical frame, generate "
            "the trim curve from anatomical landmarks, and write the ordered "
            "curve points out."
        ),
    )
    fit.add_argument("scan", type=Path, help="input limb scan (STL/PLY)")
    fit.add_argument(
        "-o",
        "--out",
        type=Path,
        help=(
            "output path; format inferred from suffix (.json/.csv/.npy for the "
            "ordered curve points, .step/.stp for a CAD/CAM-ready STEP polyline). "
            "Defaults to '<scan-stem>.trimline.json' beside the scan."
        ),
    )
    fit.add_argument(
        "--landmarks",
        type=Path,
        help=(
            "JSON sidecar mapping landmark name -> [x, y, z]. Coordinates are "
            "interpreted in the CANONICAL frame (placed on the NORMALIZED mesh: "
            "long axis +Z, proximal rim at Z=0, origin at the rim centroid), not "
            "raw scanner coordinates. Each landmark is snapped to the nearest "
            "normalized-mesh vertex for robustness. If omitted, the interactive "
            "picker would run -- but that is not implemented yet (#6), so pass "
            "this for now."
        ),
    )

    sub.add_parser("view", help="open a scan in the viewer").add_argument(
        "scan", type=Path, help="input limb scan (STL/PLY)"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        return _run_fit(args)
    if args.command == "view":
        return _run_view(args)
    # argparse(required=True) prevents reaching here, but be explicit.
    print(f"Unknown command {args.command!r}.")
    return 2


def _run_fit(args: argparse.Namespace) -> int:
    """Run load -> normalize -> landmarks -> generate -> write. Returns exit code."""
    # --- Load + normalize ----------------------------------------------------
    try:
        mesh = io.load_mesh(args.scan)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    normalized = io.normalize_orientation(mesh)

    # --- Obtain landmarks (canonical frame) ----------------------------------
    if args.landmarks is not None:
        try:
            landmarks = load_landmarks_json(args.landmarks, mesh=normalized)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1
    else:
        try:
            landmarks = place_landmarks(normalized)
        except NotImplementedError:
            print(
                "Error: the interactive landmark picker is not implemented yet "
                "(#6). Pass --landmarks PATH with a JSON sidecar of landmark "
                "name -> [x, y, z] in the canonical frame (placed on the "
                "normalized mesh) to run the pipeline headless."
            )
            return 1

    missing = landmarks.missing()
    if missing:
        names = ", ".join(lm.value for lm in missing)
        print(f"Error: landmark sidecar is missing required landmarks: {names}.")
        return 1

    # --- Generate the trim curve ---------------------------------------------
    try:
        curve = generate_trimline(normalized, landmarks)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # --- Resolve output path + write -----------------------------------------
    out_path = args.out if args.out is not None else _default_out(args.scan)
    try:
        written = _write_curve(curve, out_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # --- Summary + validation metrics ----------------------------------------
    metrics = validate_trimline(curve, normalized)
    print(f"Trim line: {len(curve)} points -> {written}")
    print(
        "Metrics: "
        f"closure_gap={metrics['closure_gap']:.3f} mm, "
        f"max_distance_to_surface={metrics['max_distance_to_surface']:.3f} mm, "
        f"self_intersections={metrics['self_intersections']}"
    )
    _warn_on_metrics(metrics)
    return 0


def _run_view(args: argparse.Namespace) -> int:
    """The viewer subcommand. Still a stub (#7); fail cleanly, no traceback."""
    print(
        "Error: the interactive viewer is not implemented yet (#7). Use 'fit' "
        "with --landmarks to run the headless pipeline for now."
    )
    return 1


def _default_out(scan: Path) -> Path:
    """Default curve output beside the scan: '<scan-stem>.trimline.json'."""
    return scan.with_name(f"{scan.stem}.trimline.json")


def _write_curve(curve: np.ndarray, path: Path) -> Path:
    """Write the ordered curve points; format inferred from the suffix.

    ``.json`` -> list of ``[x, y, z]``; ``.csv`` -> header + rows; ``.npy`` ->
    raw array. ``.step``/``.stp`` route to :func:`io.export_curve`, which writes a
    CAD/CAM-ready STEP polyline.

    Raises:
        ValueError: The suffix is not a supported output format.
    """
    suffix = path.suffix.lower()
    if suffix in _STEP_SUFFIXES:
        return io.export_curve(curve, path)
    if suffix not in _POINT_SUFFIXES:
        supported = ", ".join(sorted(_POINT_SUFFIXES | _STEP_SUFFIXES))
        raise ValueError(
            f"Unsupported output format {suffix or '<no suffix>'!r} for {path.name}; "
            f"expected one of: {supported}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        path.write_text(json.dumps(curve.tolist(), indent=2))
    elif suffix == ".csv":
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "z"])
            writer.writerows(curve.tolist())
    elif suffix == ".npy":
        np.save(path, curve)
    return path


def _warn_on_metrics(metrics: dict) -> None:
    """Print warnings (not errors) if validation metrics look off."""
    if metrics["self_intersections"] > 0:
        print(
            f"Warning: trim loop has {metrics['self_intersections']} self-intersection(s) "
            "in the XY projection; review before trusting for a socket."
        )
    if metrics["max_distance_to_surface"] > 1.0:
        print(
            f"Warning: max distance to surface is {metrics['max_distance_to_surface']:.3f} mm; "
            "the loop may not sit cleanly on the mesh."
        )


if __name__ == "__main__":
    raise SystemExit(main())
