# PyTorch fitting through native responses

`uv sync --locked` installs PyTorch with the project environment.
These four Python-only programs contain their own setup,
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
explained in [fitting examples](../fitting/index.md).

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Volume material with L-BFGS](./volume-equilibrium-material-fit-torch.md). Volume material recovery with native autograd.
- [Shell material with L-BFGS](./shell-equilibrium-material-fit-torch.md). Shell material recovery with native autograd.
- [Volume plastic stretches with L-BFGS](./volume-equilibrium-fp-fit-torch.md). Three-stretch inverse equilibrium with L-BFGS.
- [Shell plastic stretches with L-BFGS](./shell-equilibrium-fp-fit-torch.md). Two-stretch shell inverse equilibrium with L-BFGS.
