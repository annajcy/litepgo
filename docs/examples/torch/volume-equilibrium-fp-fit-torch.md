# Volume plastic stretches with L-BFGS

[Back to PyTorch fitting](./index.md)

**Experiment.** Fix material `(2,3)` on the tet bar; prescribe top displacement
$(0.04H,0,0.05H)$ with the bottom fixed. Fit
$F_p=\operatorname{diag}(e^{p_0},e^{p_1},e^{p_2})$ from identity to stretches
`(1.04,0.97,1.02)`. Observations combine free displacement and constrained
reactions with separate fixed target-norm scales.

**API walkthrough.** Select plastic-force Jacobian columns 0/4/8 and apply the
log-stretch factors inside the parameter derivative hook of `PlasticModel`.
`StaticEquilibriumResponse` consumes that parameterized energy model.
The observation Jacobian stacks $U_f$ and $(B+KU)_c$. The Torch adapter
then applies the scaled Jacobian transpose in each L-BFGS backward call.

**Expected result.** Correct three-stretch recovery and held-out observations
at load factor 0.6. Checks include re-solved finite differences, local rank at
initial/true/fitted states, and nonzero state sensitivity. This is restricted
plastic-state identification; no yield or evolution law is differentiated.

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_fp_fit_torch.py
```

::: details Complete Python source — `volume_equilibrium_fp_fit_torch.py`

<<< ../../../examples/fem/fitting/volume_equilibrium_fp_fit_torch.py

:::
