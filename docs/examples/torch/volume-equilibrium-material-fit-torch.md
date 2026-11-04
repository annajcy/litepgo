# Volume material with L-BFGS

[Back to PyTorch fitting](./index.md)

**Experiment.** On the bundled tet bar, observe the summed top reactions from
x-shear $0.08H$ and z-compression $-0.04H$, bottom fixed and free nodes
relaxed. Material is Neo-Hookean with $F_p=I$. Optimize log material
parameters $\theta=e^p$, starting at `(1,1)` in physical parameters, to recover
truth `(2,3)`. Each reaction is scaled by its fixed observed magnitude.

**API walkthrough.** `StaticEquilibriumResponse` consumes a `ParametricEnergy` model
whose hooks use `VolumetricFemEnergy` and analytic material-force partials. The outer residual
sums reaction derivatives and applies the exponential chain rule.
`lp_torch.response` connects this residual to L-BFGS's closure. Its autograd
parameter is the two-entry log vector.

**Expected result.** Recover both material parameters and predict reactions on
held-out z-compression $-0.07H$. Printed diagnostics include recovered
parameters and fitting/prediction errors. This changes the outer optimizer from
the native material experiment; the inner equilibrium and analytic derivative
model are the same. It does not train displacement tensors or material fields
through the analytic model hooks.

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit_torch.py
```

::: details Complete Python source — `volume_equilibrium_material_fit_torch.py`

<<< ../../../examples/fem/fitting/volume_equilibrium_material_fit_torch.py

:::
