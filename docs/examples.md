# Example catalog

Native cases have matching `.cpp` and `.py` files; the optional Torch fitting
examples are Python-only. Each example keeps its setup and solve in its own file.
The CMake target matches
the filename stem. Executables mirror source directories under `build/<preset>/`.
Run Python examples from the configured environment (`.venv/bin/python` or
`PYTHONPATH=build/<preset>/python .venv/bin/python`).

| Directory | What to learn | Starting case |
| --- | --- | --- |
| [energy/scalar/](https://github.com/annajcy/litepgo/tree/dev/examples/energy/scalar/) | Implement and combine scalar energies | `custom_energy`, `energy_composition` |
| [energy/response/](https://github.com/annajcy/litepgo/tree/dev/examples/energy/response/) | Vector responses, Jacobians and batched force fitting | `custom_response`, `mesh_response` |
| [solver/](https://github.com/annajcy/litepgo/tree/dev/examples/solver/) | Newton and Gauss–Newton on small problems | `newton_chain`, `gauss_newton_fit` |
| [fem/equilibrium/](https://github.com/annajcy/litepgo/tree/dev/examples/fem/equilibrium/) | Elastic static equilibrium | `tet_p1_corner_pull`, `shell_saddle_pull` |
| [fem/plasticity/](https://github.com/annajcy/litepgo/tree/dev/examples/fem/plasticity/) | Equilibrium with prescribed plastic fields | `tet_p1_plastic_twist`, `shell_plastic_field` |
| [fem/fitting/](https://github.com/annajcy/litepgo/tree/dev/examples/fem/fitting/) | Material and plastic-state inverse problems | `volume_equilibrium_material_fit`, `shell_equilibrium_fp_fit` |
| [fem/embedding/](https://github.com/annajcy/litepgo/tree/dev/examples/fem/embedding/) | Point location and displacement interpolation | `point_embedding` |
| [contact/](https://github.com/annajcy/litepgo/tree/dev/examples/contact/) | FEM with contact | `tet_p1_box_stack` |
| [mesh/](https://github.com/annajcy/litepgo/tree/dev/examples/mesh/) | Mesh optimization | `arap`, `laplacian_optimization` |
| [spatial/](https://github.com/annajcy/litepgo/tree/dev/examples/spatial/) | Spatial-index queries | `spatial_query` |

In FEM fitting names, `fixed_pose` means the displacement is supplied and held
fixed; `equilibrium` means every parameter trial solves for displacement again.
`material` fits constitutive parameters, while `fp` fits plastic stretches.
Volume and shell names distinguish the model; basis-specific names distinguish
Tet P1, Hex Q1 and Hex Hermite forward examples.

```sh
cmake --build build/release --target shell_equilibrium_fp_fit
build/release/examples/fem/fitting/shell_equilibrium_fp_fit
.venv/bin/python examples/fem/fitting/shell_equilibrium_fp_fit.py
```

See [FEM details](https://github.com/annajcy/litepgo/blob/dev/examples/fem/README.md), [energy APIs](https://github.com/annajcy/litepgo/blob/dev/examples/energy/README.md),
[solver examples](https://github.com/annajcy/litepgo/blob/dev/examples/solver/README.md), and [mesh optimization](https://github.com/annajcy/litepgo/blob/dev/examples/mesh/README.md).
