# PyTorch fitting through native responses

Install the optional dependency with `uv pip install torch` after creating the
project environment. These four Python-only programs contain their own setup,
residual, optimizer closure, and checks. They use CPU float64 one-dimensional
tensors and do not import another example.

## Objective and backward rule

Each experiment builds a native differentiable observation $y(p)$, with
fixed training scale $S$, and minimizes

$$
r(p)=S^{-1}(y(p)-y_{\rm obs}),\qquad
\mathcal L(p)=\tfrac12\|r(p)\|^2,\qquad
\nabla_p\mathcal L=J_r^Tr.
$$

`litepgo.torch.response(residual, parameters)` evaluates native values and saves
its analytic sparse Jacobian for backward. A closure calls `zero_grad`, evaluates
the loss, then `backward`; `torch.optim.LBFGS` can call it repeatedly during line
search. The native equilibrium sensitivity uses
$K_{ff}U_f=-B_f$, and reaction derivatives use $B+KU$.
No unrolled differentiation of Newton iterations is performed.

Only the explicit tensor input is differentiable. Model captures remain fixed.
There is no GPU, tensor batch dimension, or double-backward support. These are
noise-free recovery tests, not general optimizers with statistical guarantees.
They print diagnostics and write no meshes. The matching native formulations are
explained in [fitting examples](fitting.md).

All commands run from the repository root after [configuring the release build](../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Volume material with L-BFGS

**Experiment.** On the bundled tet bar, observe the summed top reactions from
x-shear $0.08H$ and z-compression $-0.04H$, bottom fixed and free nodes
relaxed. Material is Neo-Hookean with $F_p=I$. Optimize log material
parameters $\theta=e^p$, starting at `(1,1)` in physical parameters, to recover
truth `(2,3)`. Each reaction is scaled by its fixed observed magnitude.

**API walkthrough.** `StaticEquilibriumResponse` composes a
`VolumetricFemEnergy` factory with material-force partials. The outer residual
sums reaction derivatives and applies the exponential chain rule.
`lp_torch.response` connects this residual to L-BFGS's closure. Its autograd
parameter is the two-entry log vector.

**Expected result.** Recover both material parameters and predict reactions on
held-out z-compression $-0.07H$. Printed diagnostics include recovered
parameters and fitting/prediction errors. This changes the outer optimizer from
the native material experiment; the inner equilibrium and analytic derivative
model are the same. It does not train displacement tensors or material fields
through Python callback internals.

### Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit_torch.py
```

::: details Complete Python source — `volume_equilibrium_material_fit_torch.py`

<<< ../../examples/fem/fitting/volume_equilibrium_material_fit_torch.py

:::


## Shell material with L-BFGS

**Experiment.** Use the generated eight-triangle, nine-vertex shell patch with
fixed thickness 0.1. Boundary displacements combine nonuniform stretch, shear,
and bending; the center vertex relaxes. Observe its 24 boundary reaction
components at plane-stress truth `(2,3)`, scaled by their target norm. Optimize
material parameters directly from `(0.7,1.2)`. Invalid parameter trials raise an
error; the adapter does not retry them on behalf of L-BFGS.

**API walkthrough.** `ShellFemEnergy` and `ShellFemForceMaterialResponse` supply
the equilibrium factory and fixed-pose partial. `StaticEquilibriumResponse`
provides state sensitivity, and the residual selects fixed rows of $B+KU$.
L-BFGS differentiates `0.5 * lp_torch.response(...).square().sum()` through the
native Jacobian.

**Expected result.** Recover `(2,3)`, small residual and gradient, and correct
predictions at load factor 0.6. The program also checks finite-difference
state/reaction derivatives, local identifiability, fixed values, and equilibrium.
Geometry and thickness remain fixed. The [native shell material case](fitting.md#equilibrium-shell-material)
states the boundary field explicitly; its complete definition is also in the
source below.

### Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_material_fit_torch.py
```

::: details Complete Python source — `shell_equilibrium_material_fit_torch.py`

<<< ../../examples/fem/fitting/shell_equilibrium_material_fit_torch.py

:::


## Volume plastic stretches with L-BFGS

**Experiment.** Fix material `(2,3)` on the tet bar; prescribe top displacement
$(0.04H,0,0.05H)$ with the bottom fixed. Fit
$F_p=\operatorname{diag}(e^{p_0},e^{p_1},e^{p_2})$ from identity to stretches
`(1.04,0.97,1.02)`. Observations combine free displacement and constrained
reactions with separate fixed target-norm scales.

**API walkthrough.** Select plastic-force Jacobian columns 0/4/8 and apply the
log-stretch factors before passing the partial to `StaticEquilibriumResponse`.
The observation Jacobian stacks $U_f$ and $(B+KU)_c$. The Torch adapter
then applies the scaled Jacobian transpose in each L-BFGS backward call.

**Expected result.** Correct three-stretch recovery and held-out observations
at load factor 0.6. Checks include re-solved finite differences, local rank at
initial/true/fitted states, and nonzero state sensitivity. This is restricted
plastic-state identification; no yield or evolution law is differentiated.

### Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/volume_equilibrium_fp_fit_torch.py
```

::: details Complete Python source — `volume_equilibrium_fp_fit_torch.py`

<<< ../../examples/fem/fitting/volume_equilibrium_fp_fit_torch.py

:::


## Shell plastic stretches with L-BFGS

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

### Run and inspect

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/shell_equilibrium_fp_fit_torch.py
```

::: details Complete Python source — `shell_equilibrium_fp_fit_torch.py`

<<< ../../examples/fem/fitting/shell_equilibrium_fp_fit_torch.py

:::
