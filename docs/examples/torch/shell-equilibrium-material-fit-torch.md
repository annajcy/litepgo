# Shell material with L-BFGS

[Back to PyTorch fitting](./index.md)

**Experiment.** Use the generated eight-triangle, nine-vertex shell patch with
fixed thickness 0.1. Boundary displacements combine nonuniform stretch, shear,
and bending; the center vertex relaxes. Observe its 24 boundary reaction
components at plane-stress truth `(2,3)`, scaled by their target norm. Optimize
material parameters directly from `(0.7,1.2)`. Invalid parameter trials raise an
error; the adapter does not retry them on behalf of L-BFGS.

**Shared model.** This driver imports the model and residual from
[the native Python example](../fitting/shell-equilibrium-material-fit.md); that page contains the
complete setup. The source below focuses on the Torch optimizer.

**API walkthrough.** `ShellFemEnergy` and `ShellFemForceMaterialResponse` supply
the energy and fixed-pose response factories. `StaticEquilibriumResponse`
provides state sensitivity, and the residual selects fixed rows of $B+KU$.
L-BFGS differentiates `0.5 * lp_torch.response(...).square().sum()` through the
native response VJP (using the residual’s explicit-Jacobian fallback).

**Expected result.** Recover `(2,3)`, small residual and gradient, and correct
predictions at load factor 0.6. Tests cover finite-difference state/reaction derivatives, local identifiability,
fixed values, and equilibrium.
Geometry and thickness remain fixed. The [native shell material case](../fitting/shell-equilibrium-material-fit.md)
states the boundary field explicitly.

## Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_material_fit_torch.py
```

::: details Complete Python source — `shell_equilibrium_material_fit_torch.py`

<<< ../../../examples/fem/fitting/shell_equilibrium_material_fit_torch.py

:::
