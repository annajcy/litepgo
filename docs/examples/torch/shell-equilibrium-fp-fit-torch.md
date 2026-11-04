# Shell plastic stretches with L-BFGS

[Back to PyTorch fitting](./index.md)

**Experiment.** On the generated shell patch, fix material `(2,3)` and thickness
0.1. Fit $F_p=\operatorname{diag}(e^{p_0},e^{p_1},1)$ from identity to
`(1.06,0.95)`. Boundary loading is the same nonuniform stretch/shear/bending field
as the shell material experiment, with the center free. Observe scaled free
motion and boundary reactions.

**Shared model.** This driver imports the model and residual from
[the native Python example](../fitting/shell-equilibrium-fp-fit.md); that page contains the
complete setup. The source below focuses on the Torch optimizer.

**API walkthrough.** `ElementwiseResponse` gathers shared log stretches and applies
`LogDiagonalFpResponse` to produce per-element physical Fp values. `ComposedResponse`
connects this transform to the native fixed-pose force response, including Jacobian,
JVP, and VJP chain rules. The energy factory uses the same transform to construct
its Fp list. `StaticEquilibriumResponse` supplies the state sensitivity; reaction
derivatives include both the fixed-state parameter derivative and the stiffness
multiplied by state sensitivity. See [parameter transforms](../../guide/energy/parameter-transforms.md).

**Expected result.** Recover the parameters and predict observations at held-out load factor 0.6.
Tests cover rank, equilibrium, and re-solved derivatives. The normal plastic stretch and rotations are
not fitted because the planar experiment cannot identify them independently.
The optimizer exposes only the two explicit log-stretch tensor entries to autograd.

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_fp_fit_torch.py
```

::: details Complete Python source — `shell_equilibrium_fp_fit_torch.py`

<<< ../../../examples/fem/fitting/shell_equilibrium_fp_fit_torch.py

:::
