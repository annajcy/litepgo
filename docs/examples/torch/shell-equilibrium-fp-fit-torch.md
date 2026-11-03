# Shell plastic stretches with L-BFGS

[Back to PyTorch fitting](./index.md)

**Experiment.** On the generated shell patch, fix material `(2,3)` and thickness
0.1. Fit $F_p=\operatorname{diag}(e^{p_0},e^{p_1},1)$ from identity to
`(1.06,0.95)`. Boundary loading is the same nonuniform stretch/shear/bending field
as the shell material experiment, with the center free. Observe scaled free
motion and boundary reactions.

**API walkthrough.** `ShellFemForceFpResponse` supplies the fixed-pose plastic
partial; columns 0/4 and exponential factors define the two log-stretch inputs.
`StaticEquilibriumResponse`, total reaction differentiation, and
`lp_torch.response` form the L-BFGS graph. Original shell integration areas and
thickness stay fixed as natural forms change.

**Expected result.** Recovery, held-out load factor 0.6, rank, equilibrium, and
re-solved derivative checks pass. The normal plastic stretch and rotations are
not fitted because the planar experiment cannot identify them independently.
The optimizer exposes only the two explicit log-stretch tensor entries to autograd.

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_fp_fit_torch.py
```

::: details Complete Python source — `shell_equilibrium_fp_fit_torch.py`

<<< ../../../examples/fem/fitting/shell_equilibrium_fp_fit_torch.py

:::
