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

`StaticEquilibriumResponse(num_parameters, fixed, initial, options, make_energy,
make_response)` implements this map from parameters to full displacement:

- `make_energy(p)` creates the **total** potential for those parameters.
- `make_response(u)` returns a `Response` mapping parameters to the total potential's
  state gradient at that fixed displacement, including parameter-dependent loads.
  It has `num_parameters` inputs and `len(initial)` outputs.
- Every evaluation starts from the same `initial` vector. Fixed values come
  from that vector. Value-only requests skip the sensitivity solve.
- Jacobian requests use the exact, unshifted $K_{ff}$, factorized once for all
  parameter columns. Projected Hessians used by forward Newton do not replace it.

```python
from litepgo import solver

equilibrium = solver.StaticEquilibriumResponse(
    num_parameters, fixed, initial, options, make_energy, make_response)
evaluation = equilibrium.evaluate(p)
du_dp = evaluation.jacobian()
u = evaluation.value
```

C++ uses the same argument order. `MakeEnergy` returns `shared_ptr<Energy>` and
`MakeResponse` returns `shared_ptr<Response>`. Both factories are retained by the
equilibrium object and exposed to Python's cyclic garbage collector. The response
factory is called at the solved state only when derivatives on free DOFs are needed.
Its evaluation prepares any needed dynamic structure at the current parameters.

Callers must ensure
`make_response(u).evaluate(p).value == make_energy(p).eval_gradient(u)` and that the
response derivatives match that value. This is not checked automatically. Keep
captured configuration fixed, copy the supplied state if retaining it, and check
analytic derivatives against finite differences in tests.

The former matrix callback `partial(p,u)` is replaced by `make_response(u)`;
`ParametricEnergy` and its views have been removed. Existing FEM force responses
can be returned directly. Custom parameterizations can use [Response transforms and composition](../energy/parameter-transforms.md).

This requires a converged isolated differentiable equilibrium and a nonsingular
free stiffness. Forward nonconvergence or sensitivity-solve failure throws;
no approximate derivative is returned. Branch switches, singular states, and
parameter-dependent prescribed boundary data are outside this contract.
Convergence does not certify mechanical stability; the sensitivity solve accepts
indefinite nonsingular stiffness. Callbacks and captured data must be deterministic.

## Directional and adjoint derivatives

`StaticEquilibriumResponse.evaluate(p).jvp(v)` solves
`K du = -B v` on the free DOFs and returns the full displacement direction,
with zeros at fixed DOFs. `evaluate(p).vjp(w)` solves the adjoint equation
`K^T lambda = w_free` and returns `-B^T lambda_full`, extending the
adjoint with zeros at fixed DOFs. Here `K` is the exact equilibrium Hessian
and `B = make_response(u).evaluate(p).jacobian()`. Since the energy Hessian is symmetric, both paths
use the existing direct stiffness solver.

Each evaluation solves from the configured initial state once and retains the
converged displacement. Products use one directional or adjoint right-hand-side
solve and reuse the exact stiffness factorization, without forming the state
Jacobian. Force response direct products also avoid assembling `B`.
The full state-Jacobian structure is allocated only for `evaluation.jacobian()`.
Gauss–Newton can request that explicit matrix through the same evaluation.

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

## Reusing an equilibrium

`state = equilibrium.evaluate(p)` binds an independent state to a copy of `p`.
`state.value` solves from the configured initial displacement once. The first
`state.jvp(v)`, `state.vjp(w)` or `state.jacobian()` prepares the force response
and exact reduced stiffness; later derivatives reuse the factorization. Exact
Newton runs reuse the converged Hessian. Projected runs compute the exact Hessian
for derivatives. Symbolic analysis is reused only when the reduced pattern matches.
No previous-iterate or shifted numeric factorization is used for derivatives.

Torch saves the evaluation in each autograd context. Composed responses retain
child evaluations, so neither route repeats the nonlinear solve in backward.
Multiple forwards followed by backwards in any order use their own states.
Retained evaluations keep the potential and factorization alive; release them when
finished to release this memory. Explicit Jacobian storage is created only when
requested, including for large parameter fields.
