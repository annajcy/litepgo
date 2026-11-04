# Equilibrium shell plastic stretches

[Back to Material and plastic fitting](./index.md)

**Experiment.** Use the generated 3-by-3 shell patch and boundary field from the
shell material experiment, thickness 0.1, and fixed material `(2,3)`. Fit
$F_p=\operatorname{diag}(e^{p_0},e^{p_1},1)$ with truth `(1.06,0.95)`, starting
at identity. Observe the free center displacement and 24 boundary reactions,
normalized by separate fixed target norms.

**API walkthrough.** `ElementwiseResponse` gathers shared log stretches and applies
`LogDiagonalFpResponse` to produce per-element physical Fp values. `ComposedResponse`
connects this transform to the native fixed-pose force response, including Jacobian,
JVP, and VJP chain rules. The energy factory uses the same transform to construct
its Fp list. `StaticEquilibriumResponse` supplies the state sensitivity; reaction
derivatives include both the fixed-state parameter derivative and the stiffness
multiplied by state sensitivity. See [parameter transforms](../../guide/energy/parameter-transforms.md).

**Expected result.** Recover `(1.06,0.95)` and predict both observation blocks
at load factor 0.6. Re-solved finite differences, rank, fixed-value, equilibrium,
and recovery checks gate success. Tight inner tolerances are used because
normalizing a small displacement observation amplifies derivative error.
Original shell areas/thickness remain fixed and no normal plastic stretch or
plastic evolution is fitted.

## Run and inspect

```sh
cmake --build build/release --target shell_equilibrium_fp_fit litepgo_python
build/release/examples/fem/fitting/shell_equilibrium_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_fp_fit.py
```

::: details Complete Python source — `shell_equilibrium_fp_fit.py`

<<< ../../../examples/fem/fitting/shell_equilibrium_fp_fit.py

:::

::: details Complete C++ source — `shell_equilibrium_fp_fit.cpp`

<<< ../../../examples/fem/fitting/shell_equilibrium_fp_fit.cpp

:::
