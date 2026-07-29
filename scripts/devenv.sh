#!/usr/bin/env bash
# Resolve the native runtime libraries that manylinux wheels (numpy, trimesh,
# and later open3d/pyvista/VTK) need at import time on NixOS (non-FHS).
#
# Why this exists: pip installs prebuilt manylinux wheels whose C-extensions are
# dynamically linked against libstdc++, libz, libgomp, libGL, etc. On a standard
# distro these live in /usr/lib; on NixOS they live in the Nix store and are not
# on the default loader path, so `import numpy` fails with "libstdc++.so.6: cannot
# open shared object file". We resolve the needed lib dirs from <nixpkgs> (no
# hardcoded store hashes, GC-safe) and prepend them to LD_LIBRARY_PATH.
#
# Usage:  source scripts/devenv.sh   (then run .venv/bin/python / pytest / ruff)
#    or:   scripts/devenv.sh <cmd...> (runs <cmd> with the env set)
set -euo pipefail

_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Libraries the wheels in this project link against. Add to this list if a new
# dependency fails to import with a "cannot open shared object file" error.
_NIX_EXPR='with import <nixpkgs> {}; lib.makeLibraryPath [
  stdenv.cc.cc.lib   # libstdc++.so.6, libgomp.so.1
  zlib               # libz.so.1
  glib               # libgthread/glib (trimesh, pyvista optional deps)
]'

_FITTING_LIBS="$(nix eval --raw --impure --expr "$_NIX_EXPR")"
export LD_LIBRARY_PATH="${_FITTING_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ "$#" -gt 0 ]; then
  exec "$@"
fi
