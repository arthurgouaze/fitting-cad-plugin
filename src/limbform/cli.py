"""Command-line entry point.

Roadmap: Week 8 (clinician wrapper).

Thin wrapper that wires the pipeline together:

    load -> clean -> normalize -> place landmarks -> generate trim line -> export

The fit subcommand runs headless when given a landmark sidecar with
--landmarks; without one it opens the interactive picker (which needs the
optional viz extra) and saves the picked sidecar for reproducibility. The
view subcommand opens the same scan, landmarks, and trim curve in the 3D
viewer, or captures them off-screen with --screenshot.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import numpy as np
import trimesh

from limbform import io, viewer
from limbform.landmarks import load_landmarks_json, place_landmarks
from limbform.trimline import TrimRules, generate_trimline, validate_trimline

# Curve output formats. Ordered-point formats are written here; .step/.stp route
# to io.export_curve, which writes a CAD/CAM-ready STEP polyline.
_POINT_SUFFIXES = {".json", ".csv", ".npy"}
_STEP_SUFFIXES = {".step", ".stp"}

# TrimRules fields exposed as fit flags: every numeric clinical parameter.
_RULE_FIELDS = [
    f for f in dataclasses.fields(TrimRules) if isinstance(f.default, (int, float))
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limbform",
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
    fit.add_argument("scan", type=Path, help="input limb scan (STL/PLY/OBJ)")
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
            "normalized-mesh vertex. If omitted, the interactive "
            "picker opens instead (needs the 'viz' extra). A sidecar is tied "
            "to the --clean setting it was placed with, since cleanup can shift "
            "the canonical frame slightly."
        ),
    )
    fit.add_argument(
        "--save-landmarks",
        type=Path,
        help=(
            "where to persist the landmarks picked interactively (canonical "
            "frame). Only used when --landmarks is omitted. Defaults to "
            "'<scan-stem>.landmarks.json' beside the scan."
        ),
    )
    _add_clean_arguments(fit)
    rules = fit.add_argument_group(
        "trim rules",
        "clinical parameters of the rule engine; defaults are uncalibrated "
        "placeholders",
    )
    for f in _RULE_FIELDS:
        rules.add_argument(
            f"--{f.name.replace('_', '-')}",
            dest=f.name,
            type=float,
            default=f.default,
            metavar="VALUE",
            help=f"TrimRules.{f.name} (default: {f.default})",
        )

    view = sub.add_parser(
        "view",
        help="open a scan in the 3D viewer",
        description=(
            "Load a limb scan, normalize it into the canonical frame, and show it "
            "in the 3D viewer with optional landmark and trim-curve overlays. "
            "Needs the optional 'viz' extra (pyvista)."
        ),
    )
    view.add_argument("scan", type=Path, help="input limb scan (STL/PLY/OBJ)")
    view.add_argument(
        "--landmarks",
        type=Path,
        help=(
            "JSON sidecar of landmark name -> [x, y, z] in the CANONICAL frame; "
            "overlaid as labelled spheres."
        ),
    )
    view.add_argument(
        "--trimline",
        action="store_true",
        help="also generate the trim curve and overlay it (requires --landmarks)",
    )
    view.add_argument(
        "--screenshot",
        type=Path,
        help="render off-screen and write this PNG instead of opening a window",
    )
    _add_clean_arguments(view)

    return parser


def _add_clean_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mesh-cleanup options shared by fit and view."""
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "repair the scan before normalization: merge duplicate vertices, "
            "drop degenerate and duplicate faces, fix winding, cap small holes, "
            "rescale metre scans to mm (default: on)"
        ),
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        metavar="N",
        help=(
            "with --clean, decimate the scan to at most N faces (needs the "
            "'decimate' extra)"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        return _run_fit(args)
    return _run_view(args)


def _run_fit(args: argparse.Namespace) -> int:
    """Run load -> clean -> normalize -> landmarks -> generate -> write.

    Returns the exit code.
    """
    normalized = _load_normalized(args)
    if normalized is None:
        return 1

    if args.landmarks is not None:
        try:
            landmarks = load_landmarks_json(args.landmarks, mesh=normalized)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1
    else:
        sidecar_out = (
            args.save_landmarks
            if args.save_landmarks is not None
            else _default_landmarks_out(args.scan)
        )
        try:
            landmarks = place_landmarks(normalized, save_path=sidecar_out)
        except ImportError as exc:
            print(
                f"Error: {exc}\n"
                "Alternatively pass --landmarks PATH with a JSON sidecar of "
                "landmark name -> [x, y, z] in the canonical frame to run the "
                "pipeline headless."
            )
            return 1
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        print(f"Landmarks: {len(landmarks.points)} placed -> {sidecar_out}")

    missing = landmarks.missing()
    if missing:
        names = ", ".join(lm.value for lm in missing)
        print(f"Error: landmark sidecar is missing required landmarks: {names}.")
        return 1

    rules = TrimRules(**{f.name: getattr(args, f.name) for f in _RULE_FIELDS})
    try:
        curve = generate_trimline(normalized, landmarks, rules=rules)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    out_path = args.out if args.out is not None else _default_out(args.scan)
    try:
        written = _write_curve(curve, out_path)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

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
    """Load, clean, and normalize a scan and show it (with overlays) in the viewer."""
    normalized = _load_normalized(args)
    if normalized is None:
        return 1

    landmarks = None
    if args.landmarks is not None:
        try:
            landmarks = load_landmarks_json(args.landmarks, mesh=normalized)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error: {exc}")
            return 1

    curve = None
    if args.trimline:
        if landmarks is None:
            print(
                "Error: --trimline needs landmarks; pass --landmarks PATH with a "
                "canonical-frame sidecar to generate the curve."
            )
            return 1
        try:
            curve = generate_trimline(normalized, landmarks)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1

    off_screen = args.screenshot is not None
    try:
        viewer.show(
            normalized,
            landmarks=landmarks,
            trimline=curve,
            off_screen=off_screen,
            screenshot=args.screenshot,
        )
    except (ImportError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    overlays = []
    if landmarks is not None:
        overlays.append(f"{len(landmarks.points)} landmark(s)")
    if curve is not None:
        overlays.append(f"trim line ({len(curve)} points)")
    summary = f"Viewed {args.scan.name}" + (f" with {', '.join(overlays)}" if overlays else "")
    print(f"{summary} -> {args.screenshot}" if off_screen else summary)
    return 0


def _load_normalized(args: argparse.Namespace) -> trimesh.Trimesh | None:
    """Load args.scan, clean it unless --no-clean, and normalize it.

    Prints the error and returns None when the scan cannot be used.
    """
    try:
        mesh = io.load_mesh(args.scan)
        if args.clean:
            mesh, report = io.clean_mesh(mesh, max_faces=args.max_faces)
            print(f"Cleanup: {report.summary()}")
        elif args.max_faces is not None:
            print("Error: --max-faces needs --clean.")
            return None
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}")
        return None
    return io.normalize_orientation(mesh)


def _default_out(scan: Path) -> Path:
    """Default curve output beside the scan: '<scan-stem>.trimline.json'."""
    return scan.with_name(f"{scan.stem}.trimline.json")


def _default_landmarks_out(scan: Path) -> Path:
    """Default picked-landmark sidecar beside the scan: '<scan-stem>.landmarks.json'."""
    return scan.with_name(f"{scan.stem}.landmarks.json")


def _write_curve(curve: np.ndarray, path: Path) -> Path:
    """Write the ordered curve points; format inferred from the suffix.

    .json -> list of [x, y, z]; .csv -> header + rows; .npy ->
    raw array. .step/.stp route to io.export_curve, which writes a
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
