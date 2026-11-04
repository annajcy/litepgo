# Equilibrium sensitivities

A static equilibrium response maps parameters to a solved displacement and,
when requested, its total parameter derivative. Fixed-pose force partials come
from [FEM force responses](../fem/force-responses.md); the state-sensitivity solve
adds the displacement change.

## Differentiate an equilibrium branch

Partition displacement into free coordinates $u_f$ and prescribed coordinates
$u_c$, where $u_c$ and the fixed index set are independent of $p$.
For total potential $E(u,p)$, write $g(u,p)=\nabla_uE(u,p)$.
An equilibrium $u^*(p)$ satisfies

$$
g_f(u^*(p),p)=0,\qquad
K=\nabla_u^2 E,\qquad B=\left.\frac{\partial g}{\partial p}\right|_u.
$$

Differentiating gives the direct sensitivity

$$
K_{ff}\frac{\partial u_f^*}{\partial p}=-B_f,\qquad
\frac{\partial u_c^*}{\partial p}=0.
$$

`StaticEquilibriumResponse(model, fixed, initial, options)` takes one
[ParametricEnergy](../energy/parametric.md) defining the total potential.

- `model.energy_at(p)` supplies the Energy for Newton.
- `model.force_at(u)` supplies the gradient and its parameter derivatives,
  including parameter-dependent loads.

- Every evaluation starts from the same `initial` vector. Fixed values come
  from that vector. Value-only requests skip the sensitivity solve.
- Jacobian requests use the exact, unshifted $K_{ff}$, factorized once for all
  parameter columns. Projected Hessians used by forward Newton do not replace it.

```python
from litepgo import solver

equilibrium = solver.StaticEquilibriumResponse(
    model, fixed, initial, options)
u, du_dp = equilibrium.eval(p, need_jacobian=True)
```

This requires a converged isolated differentiable equilibrium and a nonsingular
free stiffness. Forward nonconvergence or sensitivity-solve failure throws;
no approximate derivative is returned. Branch switches, singular states, and
parameter-dependent prescribed boundary data are outside this contract.
Convergence does not certify mechanical stability; the sensitivity solve accepts
indefinite nonsingular stiffness. Model evaluation must be deterministic.

## Directional and adjoint derivatives

`StaticEquilibriumResponse.eval_jvp(p, v)` solves
`K du = -B v` on the free DOFs and returns the full displacement direction,
with zeros at fixed DOFs. `eval_vjp(p, w)` solves the adjoint equation
`K^T lambda = w_free` and returns `-B^T lambda_full`, extending the
adjoint with zeros at fixed DOFs. Here `K` is the exact equilibrium Hessian
and `B = model.force_at(u).eval_jacobian(p)`. Since the energy Hessian is symmetric, both paths
use the existing direct stiffness solver.

Each product evaluates equilibrium from the configured initial state and uses
one derivative right-hand-side solve, without forming the numerical state
Jacobian. Calls do not cache equilibrium states or factorizations. Parameter products delegate to the model without requiring explicit `B`.
The declared state-Jacobian sparsity pattern remains allocated.
The explicit Jacobian interface and Gauss–Newton behavior are unchanged.

## Reactions and combined measurements

For a fixed-DOF selector $C$, define reaction $R=Cg(u^*(p),p)$. Its total
derivative includes both explicit parameter dependence and state change:

$$
\frac{dR}{dp}=C\left(B+K\frac{du^*}{dp}\right).
$$

Using only $CB$ would fit a fixed pose rather than the equilibrium experiment.
The plastic-equilibrium examples observe both free displacement and reaction:

$$
y(p)=\begin{bmatrix}u_f^*(p)\\Cg(u^*(p),p)\end{bmatrix},\qquad
r(p)=\begin{bmatrix}(u_f^*-u_{f,\rm obs})/s_u\\(R-R_{\rm obs})/s_R\end{bmatrix}.
$$

They fix $s_u,s_R$ from training observations, with a small floor, so the two
physical quantities have comparable influence. The scales are not recomputed
from each trial prediction.
