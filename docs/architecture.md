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
| `io.py` | Load STL/PLY, normalize to a canonical limb frame, export STL/STEP | 5, 7 |
| `landmarks.py` | Interactive anatomical landmark picking + persistence | 5 |
| `trimline.py` | Rule-based trim-curve generation + validation | 6, 7 |
| `viewer.py` | PyVista rendering + picking host | 5, 8 |
| `cli.py` | Wire the pipeline into a clinician-facing command | 8 |

## Key decisions

- **Mesh stack, not a CAD kernel** — `trimesh`/`Open3D` for geometry, `pyvista`
  for the viewer. A full kernel is only needed at the STEP-export boundary.
- **Synthetic / open meshes** during development so nothing waits on IRB.
- **Open core** — this pipeline is Apache-2.0; the managed tier is proprietary.

An architecture sketch exists alongside the planning notes (tracked separately).

## Open questions

- STEP export path: `cascadio` vs `OCP`/`build123d` vs FreeCAD headless.
- Trim-curve representation: dense polyline vs fitted spline for CAD/CAM handoff.
