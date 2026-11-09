# Hex Hermite corner pull

[Back to Elastic equilibrium](./index.md)

**Problem.** Use Q1 rest geometry and cubic Hermite displacement on
`assets/box-hex.veg`, with the same corner loads. There are 24 physical scalar
DOFs per vertex, and `GaussLegendre3D[4]` gives 64 integration points per element.

**API walkthrough.** Coordinate data uses a width-3 vertex map. Displacement
uses `VertexDofMap(topology,24)` and
`VolumetricFemEnergy[HexHermiteBasis,GaussLegendre3D[4]]`.
The energy transforms physical derivative slots to reference coefficients and
pulls derivatives back. Spring DOFs are `24*v + {0,1,2}`; all derivative slots
remain free. Read [the physical-DOF convention](../../guide/fem/hermite.md) before
changing these offsets.

**Expected result.** Converged equilibrium with a small free residual. Positive
axis-aligned affine rest boxes are required. VEG and ordinary extracted OBJ
store the displaced corners, not the full curved Hermite field; sampled-surface
output is demonstrated by the Hermite plastic-twist case.

## Run and inspect

```sh
cmake --build build/release --target hex_hermite_corner_pull litepgo_python
build/release/examples/fem/equilibrium/hex_hermite_corner_pull
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/equilibrium/hex_hermite_corner_pull.py
```

::: details Complete Python source — `hex_hermite_corner_pull.py`

<<< ../../../examples/fem/equilibrium/hex_hermite_corner_pull.py

:::

::: details Complete C++ source — `hex_hermite_corner_pull.cpp`

<<< ../../../examples/fem/equilibrium/hex_hermite_corner_pull.cpp

:::

Construct the native material with `fem.PointwiseVolumetricAdapter[fem.NeoHookean]()`
and pass it as the first argument to the FEM energy or force response.
