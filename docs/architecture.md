# Architecture

Rule-based pipeline (no ML, no full CAD kernel) for automated transtibial socket
trim-line generation.

```
 ┌──────────┐   ┌──────────────────────┐   ┌─────────────────────┐
 │  io.py   │   │     landmarks.py     │   │     trimline.py     │
 │ load +   │──▶│ clinician places     │──▶│ rule engine builds  │
 │ normalize│   │ ~6-8 landmarks (UI)  │   │ the rim curve       │
 └──────────┘   └──────────────────────┘   └──────────┬──────────┘
       ▲                  ▲                            │
       │            ┌─────┴───────┐                    ▼
       │            │  viewer.py  │            ┌───────────────┐
       └────────────┤  (pyvista)  │◀───────────┤ validate +    │
        export STL  └─────────────┘   overlay  │ export STL/STEP│
        / STEP                                 └───────────────┘
```

## Module responsibilities

| Module | Responsibility | Roadmap week |
| --- | --- | --- |
| `io.py` | Load STL/PLY/OBJ, clean, normalize to a canonical limb frame, export STL/STEP | 5, 7 |
| `landmarks.py` | Interactive anatomical landmark picking + persistence | 5 |
| `trimline.py` | Rule-based trim-curve generation + validation | 6, 7 |
| `viewer.py` | PyVista rendering + picking host | 5, 8 |
| `cli.py` | Wire the pipeline into a clinician-facing command | 8 |

Inputs are STL, PLY, or OBJ surface meshes. Before normalization, `io.clean_mesh`
merges duplicate vertices, drops degenerate and duplicate faces, fixes winding, caps small
holes (the proximal opening stays open), rescales metre scans to mm, and optionally
decimates to a face budget.

## Key decisions

- **Mesh stack, not a CAD kernel**: `trimesh` for geometry, `pyvista` (the optional `viz`
  extra) for the viewer.
- **Synthetic / open meshes** during development so nothing waits on IRB.
- **STEP without a kernel**: `step_writer.py` emits the trim loop as an AP214 millimetre
  polyline with no dependencies; the OCCT bridges were rejected as too heavy for one curve.
- **Open core**: this pipeline is AGPL-3.0-only; the managed tier is proprietary.

An architecture sketch exists alongside the planning notes (tracked separately).

## Open questions

- Trim-curve representation: dense polyline vs fitted spline for CAD/CAM handoff.
