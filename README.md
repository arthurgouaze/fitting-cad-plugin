# limbform

> Open-core CAD layer that automates the repetitive manual parts of transtibial
> prosthetic socket fitting.

The first target is automated socket trim-line generation: the rim curve a
prosthetist currently draws by hand on every socket. The workflow is rule-based:

```
limb scan (STL/PLY/OBJ)
  → clean (merge, repair, cap small holes, metre-to-mm)
  → normalize orientation
  → clinician places 6-8 anatomical landmarks
  → rule engine generates the trim curve from clinical convention
  → export CAD/CAM-ready geometry (STL for 3D print, STEP for CAD/CAM)
```

The fitting-automation core is open-source (AGPL-3.0-only) to drive adoption,
revenue is planned later from a proprietary managed tier.
The licence applies to this repository, and contributions are accepted under
the same terms.

## Status

Early build, and the pipeline runs end to end:

- **I/O:** `io.load_mesh` (STL/PLY/OBJ import with point-cloud/scene rejection),
  `io.clean_mesh` (vertex merge, degenerate/duplicate face removal, winding fix,
  small-hole capping, metre-to-mm rescale, optional decimation),
  `io.normalize_orientation` (PCA long-axis → +Z, proximal rim at origin),
  `io.export_mesh` (STL/PLY/OBJ).
- **Trim-line rule engine:** `trimline.generate_trimline` projects a
  clinician-tunable `TrimRules` height profile onto the limb surface and returns
  a closed, on-surface loop; `trimline.validate_trimline` reports closure gap,
  distance to surface, and self-intersections.
- **STEP export:** `io.export_curve` writes the trim loop as a CAD/CAM-ready
  AP214 polyline (no CAD kernel required).
- **Viewer:** `viewer.show` renders the limb, labelled landmark glyphs, and the
  trim curve; supports off-screen PNG capture.
- **Interactive landmark picker:** `landmarks.place_landmarks` walks the
  clinician through the landmark sequence with click-to-place, undo, and skip,
  snapping each pick to the nearest mesh vertex and persisting a canonical-frame
  JSON sidecar.
- **CLI:** `limbform fit` (headless with `--landmarks`, or interactive) and
  `limbform view`. Cleanup runs by default (`--no-clean` skips it,
  `--max-faces N` decimates with the `decimate` extra), and every `TrimRules`
  parameter has a `fit` flag (`--anterior-offset-mm`, `--posterior-height-mm`,
  `--smoothing`).

A committed synthetic limb mesh (`examples/synthetic_limb.stl`) backs the tests.
The viewer and picker need the optional `viz` extra
(`pip install -e '.[dev,viz]'`); everything else runs headless. Next up:
calibrating the `TrimRules` defaults against clinician-reviewed examples.

## Stack

Python mesh stack (no full CAD kernel):

- **geometry:** [`trimesh`](https://trimesh.org) + `numpy` / `scipy`, with
  `rtree` for the ray-mesh queries the rule engine uses
- **viewer / picking:** [`pyvista`](https://pyvista.org) (optional `viz` extra)
- **export:** STL / PLY / OBJ / STEP

Develop on synthetic or open limb-scan meshes as to not wait on IRB.

## Layout

```
src/limbform/
  io.py         mesh import/export, orientation normalization
  landmarks.py  landmark set, surface snapping, sidecar I/O, interactive picker
  trimline.py   rule-based trim-curve engine
  viewer.py     pyvista viewer + overlay primitives
  step_writer.py  dependency-free AP214 STEP writer
  cli.py        command-line entry point (fit / view)
tests/          pytest suite (headless; GL tests skip without the viz extra)
examples/       synthetic / open limb-scan meshes
docs/           design notes
```

## Development

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The core deps install as prebuilt manylinux wheels. On most distributions they just
work. On NixOS (non-FHS) however, the wheels' C-extensions can't find libstdc++/libz
on the default loader path, so two thin helpers resolve the native libraries
from `<nixpkgs>` (no hardcoded store paths, GC-safe):

```sh
# run anything (python / pytest) with the right LD_LIBRARY_PATH
scripts/devenv.sh .venv/bin/python -m pytest

# ruff ships a single native binary, which the following comman runs via the Nix loader
scripts/ruff.sh check .
```

On a normal FHS distro run `.venv/bin/pytest` / `.venv/bin/ruff` directly.

The interactive viewer and the landmark picker need extra GL/VTK deps, install
them with the optional `viz` extra:

```sh
.venv/bin/pip install -e '.[dev,viz]'

# generate a trim line, picking landmarks interactively (saves a sidecar)
limbform fit examples/synthetic_limb.stl -o trimline.step

# rerun headless from the sidecar with tuned rules
limbform fit examples/synthetic_limb.stl --landmarks synthetic_limb.landmarks.json \
  --anterior-offset-mm 20 --posterior-height-mm 8 -o trimline.step

# inspect a scan, its landmarks, and the generated curve
limbform view examples/synthetic_limb.stl \
  --landmarks synthetic_limb.landmarks.json --trimline
```

On NixOS the VTK wheel also dlopens GL/X11 at render time. `scripts/devenv.sh`
puts those on the loader path too, the viewer should be run through it. Add
`--screenshot out.png` to `view` to render off-screen instead of opening a
window.
