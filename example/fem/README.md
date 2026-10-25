# FEM with MeshEnergy

## Hex Q1

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

## Tet P1

Two tets share a face, with volumes `1/6` and `1/3`. They use `Interpolation<TetP1Basis>`, `DeformationGradientField<TetP1Basis>`, and `integrate_volume<TetCentroidQuadrature>` with the same `MeshEnergy` lambda pattern. Vertex order follows `(0,0,0), (1,0,0), (0,1,0), (0,0,1)` and must give a positive rest Jacobian.

The centroid rule has one point at `(1/4,1/4,1/4)` and weight `1/6`. With P1 geometry and displacement, `F` is constant per element, so the rule is exact for fixed `Fp` and `theta`. General spatially varying fields may require a higher-order rule. This example also evaluates energy and derivatives only.

```sh
cmake --build --preset debug --target tet_p1_mesh_energy litepgo_tests
./build/debug/example/fem/tet_p1_mesh_energy
./build/debug/tests/cpp/litepgo_tests --gtest_filter='TetP1Example.*'
```

The FD test in `tests/example/fem` shares the example setup and checks all 15 displacement DOFs at rest, under uniform stretch, and with nonuniform displacement.

## Hex Hermite

Two boxes of sizes `(1,2,3)` and `(2,2,3)` share a face. Geometry uses Hex Q1; displacement uses 64 Hermite coefficients per component. Each vertex has 24 physical DOFs: the three components of `u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ`.

The lambda converts physical derivatives to reference coefficients using the rest Q1 Jacobian, integrates with `GaussLegendre3D<4>`, then returns `S^T g` and `S^T H S`. Shared vertex DOFs enforce C1 continuity on conforming axis-aligned boxes in consistent Hex corner order. Other geometries are rejected. No interface changes are needed.

```sh
cmake --build --preset debug --target hex_hermite_mesh_energy litepgo_tests
./build/debug/example/fem/hex_hermite_mesh_energy
./build/debug/tests/cpp/litepgo_tests --gtest_filter='HexHermiteExample.*'
```

Tests in `tests/example/fem` check affine fields, shared-face values and physical gradients, and energy derivatives with respect to physical DOFs. The example evaluates energy and derivatives only; feasible step bounds are not implemented. Nonlinear material integration is numerical.
