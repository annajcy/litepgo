# Volume plastic stretches with L-BFGS

[Back to PyTorch fitting](./index.md)

**Experiment.** Fix material `(2,3)` on the tet bar; prescribe top displacement
$(0.04H,0,0.05H)$ with the bottom fixed. Fit
$F_p=\operatorname{diag}(e^{p_0},e^{p_1},e^{p_2})$ from identity to stretches
`(1.04,0.97,1.02)`. Observations combine free displacement and constrained
reactions with separate fixed target-norm scales.

**Shared model.** This driver imports the model and residual from
[the native Python example](../fitting/volume-equilibrium-fp-fit.md); that page contains the
complete setup. The source below focuses on the Torch optimizer.

**API walkthrough.** `ElementwiseResponse` gathers shared log stretches and applies
`LogDiagonalFpResponse` to produce per-element physical Fp values. `ComposedResponse`
connects this transform to the native fixed-pose force response, including Jacobian,
JVP, and VJP chain rules. The energy factory uses the same transform to construct
its Fp list. `StaticEquilibriumResponse` supplies the state sensitivity; reaction
derivatives include both the fixed-state parameter derivative and the stiffness
multiplied by state sensitivity. See [parameter transforms](../../guide/energy/parameter-transforms.md).

**Expected result.** Correct three-stretch recovery and held-out observations
at load factor 0.6. Tests include re-solved finite differences, local rank at
initial/true/fitted states, and nonzero state sensitivity. This is restricted
plastic-state identification; no yield or evolution law is differentiated.

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_fp_fit_torch.py
```

::: details Complete Python source — `volume_equilibrium_fp_fit_torch.py`

<<< ../../../examples/fem/fitting/volume_equilibrium_fp_fit_torch.py

:::
