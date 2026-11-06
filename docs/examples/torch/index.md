# PyTorch materials and fitting

`uv sync --locked` installs PyTorch with the project environment.
The constitutive, force-response and shell-fit examples implement W in Torch as
[Python materials](../../guide/fem/python-materials.md) and use native assembly and
equilibrium solves. The four L-BFGS drivers demonstrate the outer optimizer bridge.
These four Python-only drivers import the physical setup and residual from
the adjacent native Python fitting example. They show the Torch-specific part:
CPU float64 parameter tensors, the L-BFGS closure, and backward. Keep both
files together when copying a driver. Numerical checks live in the test suite.

For a learned constitutive model, start with the
[neural Neo-Hookean example](./neural-neohookean.md): ordinary Torch optimizers
train network weights on energy/stress samples, then native FEM evaluates the
frozen model. Its optional directional tangent comparison illustrates both the
accuracy gains and the remaining componentwise errors.

## Objective and backward rule for the four fitting drivers

Each of the four L-BFGS fitting drivers builds a native differentiable observation $y(p)$, with
fixed training scale $S$, and minimizes

$$
r(p)=S^{-1}(y(p)-y_{\rm obs}),\qquad
\mathcal L(p)=\tfrac12\|r(p)\|^2,\qquad
\nabla_p\mathcal L=J_r^Tr.
$$

`litepgo.torch.response(residual, parameters)` evaluates native values and saves
the input for backward, which calls the residual VJP. The four L-BFGS fitting drivers implement
explicit residual Jacobians, so their VJP uses the default Jacobian fallback. A closure calls `zero_grad`, evaluates
the loss, then `backward`; `torch.optim.LBFGS` can call it repeatedly during line
search. The native equilibrium sensitivity uses
$K_{ff}U_f=-B_f$, and reaction derivatives use $B+KU$.
No unrolled differentiation of Newton iterations is performed.

For these native-response optimizer bridges, only the explicit tensor input is differentiable. Model captures remain fixed.
There is no GPU, tensor batch dimension, or double-backward support. These are
noise-free recovery tests, not general optimizers with statistical guarantees.
They print diagnostics and write no meshes. The matching native formulations are
explained in [fitting examples](../fitting/index.md).

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Learn a neural material](./neural-neohookean.md). Train an invariant energy on native Neo-Hookean samples and evaluate stress, tangent, and FEM equilibrium.

- [Force responses through equilibrium](./torch-equilibrium-response.md). Torch-material force Jacobians and equilibrium JVP/VJP for physical theta and Fp.

- [Constitutive energy in Torch](./torch-neohookean.md). Write W in Torch and compare FEM derivatives/equilibrium with native Neo-Hookean on tets and hexes.

- [Shell material written in Torch](./torch-shell-material-fit.md). A Torch Koiter density fitted from equilibrium reactions with Gauss–Newton.

- [Volume material with L-BFGS](./volume-equilibrium-material-fit-torch.md). Volume material recovery with native autograd.
- [Shell material with L-BFGS](./shell-equilibrium-material-fit-torch.md). Shell material recovery with native autograd.
- [Volume plastic stretches with L-BFGS](./volume-equilibrium-fp-fit-torch.md). Three-stretch inverse equilibrium with L-BFGS.
- [Shell plastic stretches with L-BFGS](./shell-equilibrium-fp-fit-torch.md). Two-stretch shell inverse equilibrium with L-BFGS.
