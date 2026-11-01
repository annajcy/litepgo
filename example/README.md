# Example catalog

Native cases have matching `.cpp` and `.py` files; the optional Torch fitting
examples are Python-only. Each example keeps its setup and solve in its own file.
The CMake target matches
the filename stem. Executables mirror source directories under `build/<preset>/`.
Run Python examples from the configured environment (`.venv/bin/python` or
`PYTHONPATH=build/<preset>/python .venv/bin/python`).

| Directory | What to learn | Starting case |
| --- | --- | --- |
| [energy/scalar/](energy/scalar/) | Implement and combine scalar energies | `custom_energy`, `energy_composition` |
| [energy/response/](energy/response/) | Vector responses, Jacobians and batched force fitting | `custom_response`, `mesh_response` |
| [solver/](solver/) | Newton and Gauss–Newton on small problems | `newton_chain`, `gauss_newton_fit` |
| [fem/equilibrium/](fem/equilibrium/) | Elastic static equilibrium | `tet_p1_corner_pull`, `shell_saddle_pull` |
| [fem/plasticity/](fem/plasticity/) | Equilibrium with prescribed plastic fields | `tet_p1_plastic_twist`, `shell_plastic_field` |
| [fem/fitting/](fem/fitting/) | Material and plastic-state inverse problems | `volume_equilibrium_material_fit`, `shell_equilibrium_fp_fit` |
| [fem/embedding/](fem/embedding/) | Point location and displacement interpolation | `point_embedding` |
| [contact/](contact/) | FEM with contact | `tet_p1_box_stack` |
| [mesh/](mesh/) | Mesh optimization | `arap`, `laplacian_optimization` |
| [spatial/](spatial/) | Spatial-index queries | `spatial_query` |

In FEM fitting names, `fixed_pose` means the displacement is supplied and held
fixed; `equilibrium` means every parameter trial solves for displacement again.
`material` fits constitutive parameters, while `fp` fits plastic stretches.
Volume and shell names distinguish the model; basis-specific names distinguish
Tet P1, Hex Q1 and Hex Hermite forward examples.

```sh
cmake --build build/release --target shell_equilibrium_fp_fit
build/release/example/fem/fitting/shell_equilibrium_fp_fit
.venv/bin/python example/fem/fitting/shell_equilibrium_fp_fit.py
```

See [FEM details](fem/README.md), [energy APIs](energy/README.md),
[solver examples](solver/README.md), and [mesh optimization](mesh/README.md).
