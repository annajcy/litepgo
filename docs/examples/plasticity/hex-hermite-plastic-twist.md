# Hex Hermite plastic twist

[Back to Prescribed plastic fields](./index.md)

**Problem and discretization.** On `assets/box-hex.veg`, supply the twist's
values and physical derivatives in all eight Hermite vector slots. Q1 rest
geometry and `GaussLegendre3D[4]` integration are used. Both equilibrium and the
compatible plastic displacement belong to the same Hermite space.

**API walkthrough.** `VertexDofMap(topology,24)` gathers physical DOFs.
`BasisDofTransform[HexHermiteBasis]` converts plastic slots to reference
coefficients before constructing `DeformationGradientField3D`. End clamps fix
slots 0, 1, 2, and 4: $u,u_{,X},u_{,Y},u_{,XY}$, which determine a face
at constant $Z$. Fixing value slots alone would not fix that face field.

For surface output, `sample_boundary[HexQ1Basis](...,4)` generates reference
samples and `build_point_displacement_map[HexHermiteBasis]` evaluates the solved
field there. The OBJ therefore includes derivative-DOF contributions between
corners, with 32 triangles per boundary quad.

**Expected result.** Released and clamped equilibria preserve fixed physical
slots. The free case relaxes toward the compatible plastic shape. VEG stores
only displaced corners; use the sampled OBJ to inspect the Hermite boundary.
Rest boxes must be positive, affine, and axis-aligned. A Q1 plastic interpolant
would generally be incompatible with C1 Hermite displacement across faces.

## Run and inspect

```sh
cmake --build build/release --target hex_hermite_plastic_twist litepgo_python
build/release/examples/fem/plasticity/hex_hermite_plastic_twist
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/plasticity/hex_hermite_plastic_twist.py
```

::: details Complete Python source — `hex_hermite_plastic_twist.py`

<<< ../../../examples/fem/plasticity/hex_hermite_plastic_twist.py

:::

::: details Complete C++ source — `hex_hermite_plastic_twist.cpp`

<<< ../../../examples/fem/plasticity/hex_hermite_plastic_twist.cpp

:::
