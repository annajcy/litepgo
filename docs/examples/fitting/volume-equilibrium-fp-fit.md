# Equilibrium volume plastic stretches

[Back to Material and plastic fitting](./index.md)

**Experiment.** Use `assets/box-tet.veg`, fixed material `(2,3)`, bottom fixed,
and top displaced by $(0.04H,0,0.05H)$. Fit three shared positive diagonal
plastic stretches with truth `(1.04,0.97,1.02)`, starting from identity in log
coordinates. Observe both free displacements and fixed-DOF reactions at equilibrium.
Each observation block is divided by its fixed target norm, floored at $10^{-8}$.

**API walkthrough.** `VolumetricFemForceFpResponse` supplies the nine-column
fixed-pose derivative. Select columns 0/4/8 and apply $e^p$ to obtain $B_p$.
`StaticEquilibriumResponse(3,...)` solves $K_{ff}U_f=-B_{p,f}$.
Stack free rows of $U$ and fixed rows of $B_p+KU$ to form the total
observation Jacobian for the outer `FixedPatternResponse`.

**Expected result.** Recover all three stretches and predict displacement and
reaction observations at load factor 0.6. The program checks full column rank
at initial/truth/fitted states, nonzero state sensitivity, fixed values, and
re-solved finite differences of both observation blocks. The model fixes
plastic principal axes and does not assume a full nine-entry matrix is identifiable.

## Run and inspect

```sh
cmake --build build/release --target volume_equilibrium_fp_fit litepgo_python
build/release/examples/fem/fitting/volume_equilibrium_fp_fit
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_fp_fit.py
```

::: details Complete Python source — `volume_equilibrium_fp_fit.py`

<<< ../../../examples/fem/fitting/volume_equilibrium_fp_fit.py

:::

::: details Complete C++ source — `volume_equilibrium_fp_fit.cpp`

<<< ../../../examples/fem/fitting/volume_equilibrium_fp_fit.cpp

:::
