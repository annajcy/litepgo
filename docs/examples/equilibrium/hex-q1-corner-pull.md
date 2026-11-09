# Hex Q1 corner pull

[Back to Elastic equilibrium](./index.md)

**Problem.** Apply the same $t_i=0.3(X_i-X_{\min})$, $k=1$ loads to
`assets/box-hex.veg`. Geometry and displacement are trilinear Q1; integration uses
`GaussLegendre3D[2]`, eight points per element.

**API walkthrough.** Use `VertexDofMap(topology,3)` and Q1 rest fields, then
`VolumetricFemEnergy[HexQ1Basis,GaussLegendre3D[2]]`.
The spring composition, empty hard-constraint set, and Newton call match the tet
case. Keep geometry fields in element connectivity order.

**Expected result.** A converged pulled box with valid deformed meshes and a
small free residual. Unlike Tet P1, the quadrature-point determinant checks are
not a proof of injectivity everywhere inside a general trilinear element.
The output surface connects deformed corner positions.

## Run and inspect

```sh
cmake --build build/release --target hex_q1_corner_pull litepgo_python
build/release/examples/fem/equilibrium/hex_q1_corner_pull
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/equilibrium/hex_q1_corner_pull.py
```

::: details Complete Python source — `hex_q1_corner_pull.py`

<<< ../../../examples/fem/equilibrium/hex_q1_corner_pull.py

:::

::: details Complete C++ source — `hex_q1_corner_pull.cpp`

<<< ../../../examples/fem/equilibrium/hex_q1_corner_pull.cpp

:::

Construct the native material with `fem.PointwiseVolumetricAdapter[fem.NeoHookean]()`
and pass it as the first argument to the FEM energy or force response.
