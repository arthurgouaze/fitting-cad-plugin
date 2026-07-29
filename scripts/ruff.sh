#!/usr/bin/env bash
# Run the ruff binary from .venv on NixOS.
#
# Why: ruff's wheel ships a single dynamically-linked native binary built for
# generic Linux. NixOS has no /lib64/ld-linux loader, so executing it directly
# fails with "Could not start dynamically linked executable". We invoke the Nix
# glibc dynamic loader explicitly and provide the runtime libs via devenv.sh.
#
# Usage: scripts/ruff.sh check .   |   scripts/ruff.sh format .
set -euo pipefail
_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_LOADER="$(nix eval --raw --impure --expr \
  'with import <nixpkgs> {}; "${stdenv.cc.bintools.dynamicLinker}"')"
exec "$_REPO/scripts/devenv.sh" "$_LOADER" "$_REPO/.venv/bin/ruff" "$@"
