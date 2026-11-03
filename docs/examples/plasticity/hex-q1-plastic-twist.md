# Hex Q1 plastic twist

[Back to Prescribed plastic fields](./index.md)

**Problem and discretization.** Use `assets/box-hex.veg`, Q1 geometry,
Q1 interpolation of $u_p$, and `GaussLegendre3D[2]`. Unlike the tet loading
loop, this program directly constructs the final 90-degree plastic twist before
its two solves.

**API walkthrough.** Use width-3 vertex maps and Q1 coordinate/plastic fields.
The composed $F_p$ is supplied to
`VolumetricFemEnergy[NeoHookean,HexQ1Basis,GaussLegendre3D[2]]`.
`DofReduction` fixes bottom xyz entries, then bottom and top xyz entries for the
second solve. The same material snapshot is used for both.

**Expected result.** The displacement space contains the prescribed Q1
$u_p$, so the released state can be stress-free. The clamped energy is
positive. Exported VEG and OBJ display corner-interpolated geometry; they do not
establish a continuum injectivity guarantee between integration points.

## Run and inspect

```sh
cmake --build build/release --target hex_q1_plastic_twist litepgo_python
build/release/examples/fem/plasticity/hex_q1_plastic_twist
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/plasticity/hex_q1_plastic_twist.py
```

::: details Complete Python source — `hex_q1_plastic_twist.py`

<<< ../../../examples/fem/plasticity/hex_q1_plastic_twist.py

:::

::: details Complete C++ source — `hex_q1_plastic_twist.cpp`

<<< ../../../examples/fem/plasticity/hex_q1_plastic_twist.cpp

:::
