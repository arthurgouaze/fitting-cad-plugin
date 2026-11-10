"""Reproducible generator for the synthetic transtibial limb mesh.

Produces examples/synthetic_limb.stl: a closed, tapered, slightly elliptical
tube standing in for a transtibial residual limb. It is deliberately simple,
and involves no patient data, but has the properties the pipeline needs:

- a clear long axis (so PCA orientation has something to lock onto),
- a wide proximal end and a narrower, rounded distal end (so the
  proximal/distal sign convention in limbform.io.normalize_orientation
  can be exercised),
- a slight elliptical cross-section (anterior-posterior vs medial-lateral),
- a watertight surface (closed rim cap + rounded distal pole).

The mesh is generated in an intentionally off-canonical pose (long axis along
a tilted direction, away from the origin) so normalization has real work to do.

Run python examples/make_synthetic_limb.py to regenerate the committed STL.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

# Geometry parameters (millimetres; representative of a transtibial residuum).
LENGTH = 150.0  # proximal rim -> distal pole
PROX_A = 55.0  # proximal semi-axis, anterior-posterior
PROX_B = 45.0  # proximal semi-axis, medial-lateral (elliptical cross-section)
DISTAL_SCALE = 0.55  # mid-shaft taper relative to the proximal rim
N_THETA = 64  # angular resolution
N_RINGS = 40  # rings along the length (ring 0 = proximal rim, last = distal collar)


def build_synthetic_limb() -> trimesh.Trimesh:
    """Construct the synthetic limb mesh in a deliberately off-canonical pose.

    Topology (kept watertight after an STL round-trip):

    - N_RINGS rings of N_THETA vertices form the side wall. Every ring,
      including the last, keeps a non-zero radius: a ring tapered to radius 0
      collapses into zero-area triangles that trimesh and the STL round-trip
      split off, breaking watertightness. The distal end is closed by a single
      pole vertex instead.
    - A proximal rim cap fans the open socket end (ring 0) to a centre vertex.
    - A distal pole cap fans the last ring to one tip vertex below it, giving a
      rounded, closed distal pole with no degenerate faces.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    # t in [0, 1]: 0 = proximal rim, 1 = last ring (the distal collar).
    t = np.linspace(0.0, 1.0, N_RINGS)

    # Smooth taper from the full-width rim to a rounded distal end. The radial
    # profile stays strictly positive at every ring (min ~0.32 of full width) so
    # no ring collapses; the distal pole vertex closes the remaining hole.
    taper = 1.0 - (1.0 - DISTAL_SCALE) * t
    # Gentle distal rounding (a quarter-cosine easing over the last 25%) that
    # narrows the collar without ever reaching zero radius.
    ease = np.where(t > 0.75, np.cos((t - 0.75) / 0.25 * (np.pi / 2.0)), 1.0)
    radial = taper * (0.55 + 0.45 * ease)

    verts: list[np.ndarray] = []
    for ti, r in zip(t, radial):
        a = PROX_A * r
        b = PROX_B * r
        z = LENGTH * ti
        ring = np.column_stack(
            [a * np.cos(theta), b * np.sin(theta), np.full_like(theta, z)]
        )
        verts.append(ring)
    grid = np.array(verts)  # (N_RINGS, N_THETA, 3)

    faces: list[list[int]] = []

    def idx(ring: int, col: int) -> int:
        return ring * N_THETA + (col % N_THETA)

    # Side wall quads (two triangles each).
    for ri in range(N_RINGS - 1):
        for ci in range(N_THETA):
            a0, a1 = idx(ri, ci), idx(ri, ci + 1)
            b0, b1 = idx(ri + 1, ci), idx(ri + 1, ci + 1)
            faces.append([a0, b0, b1])
            faces.append([a0, b1, a1])

    points = grid.reshape(-1, 3).tolist()

    # Proximal rim cap (ring 0): close the open socket end with a fan.
    prox_center = len(points)
    points.append([0.0, 0.0, 0.0])
    for ci in range(N_THETA):
        faces.append([prox_center, idx(0, ci + 1), idx(0, ci)])

    # Distal pole cap: fan the last (non-degenerate) ring to a single tip vertex
    # placed a little beyond the collar, giving a rounded closed pole.
    distal_center = len(points)
    points.append([0.0, 0.0, LENGTH + 0.12 * LENGTH * (1.0 - DISTAL_SCALE)])
    last = N_RINGS - 1
    for ci in range(N_THETA):
        faces.append([distal_center, idx(last, ci), idx(last, ci + 1)])

    mesh = trimesh.Trimesh(
        vertices=np.array(points), faces=np.array(faces), process=True
    )
    # Both caps traverse their ring in the same direction as the adjacent side
    # wall faces, so the winding is inconsistent; io.clean_mesh repairs it.

    # Pose it off-canonical: rotate the long axis off +Z and translate away from
    # the origin, so normalize_orientation has a non-trivial job.
    tilt = trimesh.transformations.rotation_matrix(np.deg2rad(35.0), [1.0, 0.5, 0.0])
    tilt[:3, 3] = [120.0, -80.0, 60.0]
    mesh.apply_transform(tilt)
    return mesh


def main() -> Path:
    mesh = build_synthetic_limb()
    out = Path(__file__).parent / "synthetic_limb.stl"
    mesh.export(out)
    print(
        f"Wrote {out}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
        f"watertight={mesh.is_watertight}"
    )
    return out


if __name__ == "__main__":
    main()
