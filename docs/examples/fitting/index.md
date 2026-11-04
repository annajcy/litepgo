# Material and plastic-state identification

These seven programs fit synthetic noiseless observations. They print recovery,
residual, and held-out prediction errors and do not write meshes. A zero exit
means the solver converged. Recovery thresholds, finite differences, and rank
checks are enforced in the test suite. Read the formulation below
before selecting an experiment: fixed-pose force fitting and equilibrium fitting
use different derivatives.

## Shared formulation

Let $p$ be the chosen parameter vector, $E(u,p)$ elastic energy, and
$g=\nabla_uE$ the positive-gradient force. All cases minimize

$$
\Phi(p)=\tfrac12\|r(p)\|^2,\qquad r=S^{-1}(y(p)-y_{\rm obs}),
$$

with fixed training scales $S$. At a prescribed pose, $y=g(u_0,p)$ and
$J_y=B=\partial g/\partial p|_{u_0}$. At equilibrium, let $u_f$ be free
DOFs and $C$ select fixed-DOF reactions. Then

$$
g_f(u^*,p)=0,\qquad K_{ff}U_f=-B_f,\quad U_c=0,
$$

$$
\frac{d(Cg)}{dp}=C(B+KU),\quad K=\nabla_u^2E.
$$

`StaticEquilibriumResponse` computes $u^*,U$. Force-response objects supply
$B$; they do not include the state change by themselves. The exact unshifted
stiffness $K$ is used for sensitivities. Fixed indices and values are
parameter-independent.

Volumetric cases use the [Neo-Hookean solid energy](../../guide/fem/volume.md),
including $W=\det F_p\,\psi(FF_p^{-1})$. Shell cases use the
[Koiter natural-form energy](../../guide/fem/shell.md), with original area and
thickness fixed. Parameterized positive stretches $s_j=e^{p_j}$ multiply
selected derivative columns by $s_j$. `vec(Fp)` uses column-major order, so
diagonal entries occupy columns 0, 4, 8.

These are locally checked recovery experiments, not evidence that arbitrary
materials or full plastic matrices are identifiable. No noise model, spatial
regularization, or plastic evolution is added.

All commands run from the repository root after [configuring the release build](../../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Examples

- [Fixed-pose shell material](./shell-fixed-pose-material-fit.md). Shared shell material from fixed-pose forces.
- [Fixed-pose volume plastic stretches](./volume-fixed-pose-fp-fit.md). Three diagonal solid plastic stretches.
- [Fixed-pose shell plastic stretches](./shell-fixed-pose-fp-fit.md). Two in-plane shell plastic stretches.
- [Equilibrium volume material](./volume-equilibrium-material-fit.md). Material from equilibrium reaction experiments.
- [Equilibrium shell material](./shell-equilibrium-material-fit.md). Material-dependent state and reaction sensitivities.
- [Equilibrium volume plastic stretches](./volume-equilibrium-fp-fit.md). Plastic stretches from shapes and reactions.
- [Equilibrium shell plastic stretches](./shell-equilibrium-fp-fit.md). Shell natural geometry through equilibrium.
