# Hex Q1 with MeshEnergy

Two hexes of widths 1 and 2 share a face and its nodal displacement DOFs. Each `geometries[e]` interpolates rest coordinates gathered for element `e`; `MeshEnergy` supplies that same element's `local_u` to the lambda. `integrate_volume<GaussLegendre3D<2>>` integrates that geometry with eight Gauss points. `MaterialLocalEnergyF<NeoHookean>` differentiates displacement DOFs; `Fp = I` and `theta = [2, 3]` stay fixed. The example evaluates energy, gradient, and Hessian, and checks both zero rest energy and uniform-strain energy over the total volume of 3.

From the repository root:

```sh
cmake --preset debug
cmake --build --preset debug --target hex_q1_mesh_energy
./build/debug/example/fem/hex_q1_mesh_energy
```

The target is excluded from the default build. This example evaluates energy and derivatives only; supply feasible step bounds before using a Newton solver.

The FD test uses the same setup from `hex_q1_mesh_energy.hpp` and checks the full gradient and Hessian at rest, under uniform stretch, and with nonuniform displacement:

```sh
cmake --build --preset debug --target litepgo_tests
./build/debug/tests/cpp/litepgo_tests --gtest_filter='HexQ1Example.*'
```
