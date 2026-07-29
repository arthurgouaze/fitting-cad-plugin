# fitting-cad-plugin

> Open-core CAD layer that automates the repetitive manual parts of transtibial
> prosthetic socket **fitting**. Part of the **iStem** venture.

The first target is automated **socket trim-line generation** — the rim curve a
prosthetist currently draws by hand on every socket. The workflow is rule-based:

```
limb scan (STL/PLY)
  → normalize orientation
  → clinician places ~6–8 anatomical landmarks
  → rule engine generates the trim curve from clinical convention
  → export CAD/CAM-ready geometry (STL for 3D print, STEP for CAD/CAM)
```

The fitting-automation **core is open-source** (Apache-2.0) to drive adoption;
revenue is planned later from a proprietary managed tier.

## Status

Early build. The **I/O layer is implemented and tested**: `io.load_mesh`
(STL/PLY import with point-cloud/scene rejection), `io.normalize_orientation`
(PCA long-axis → +Z, proximal rim at origin), and `io.export_mesh` (STL/PLY).
A committed synthetic limb mesh (`examples/synthetic_limb.stl`) backs the tests.
Landmarks, the trim-line rule engine, the viewer, and STEP export are next.

## Stack

Python mesh stack (no full CAD kernel):

- **geometry:** [`trimesh`](https://trimesh.org), [`Open3D`](https://www.open3d.org)
- **viewer:** [`pyvista`](https://pyvista.org)
- **export:** STL / PLY / STEP

Develop on **synthetic or open limb-scan meshes** so nothing waits on IRB.

## Layout

```
src/fitting_cad/
  io.py         mesh import/export, orientation normalization
  landmarks.py  interactive anatomical landmark placement
  trimline.py   rule-based trim-curve engine
  viewer.py     pyvista viewer
  cli.py        command-line entry point
tests/          pytest stubs
examples/       synthetic / open limb-scan meshes (placeholder)
docs/           design notes
```

## Development

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The core deps install as prebuilt manylinux wheels. On most distros they just
work; **on NixOS** (non-FHS) the wheels' C-extensions can't find libstdc++/libz
on the default loader path, so two thin helpers resolve the native libraries
from `<nixpkgs>` (no hardcoded store paths, GC-safe):

```sh
# run anything (python / pytest) with the right LD_LIBRARY_PATH
scripts/devenv.sh .venv/bin/python -m pytest

# ruff ships a single native binary; this runs it via the Nix loader
scripts/ruff.sh check .
```

On a normal FHS distro you can skip the helpers and run `.venv/bin/pytest` /
`.venv/bin/ruff` directly.

The interactive viewer + landmark UI need extra GL/VTK deps; install them with
the optional `viz` extra (`pip install -e '.[dev,viz]'`). `open3d` currently has
no wheel for Python 3.13, so use a 3.11/3.12 venv if you need it.

## Project notes

Planning, roadmap, and discovery notes are tracked separately, outside this repo.
