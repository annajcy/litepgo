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
partial)` implements this map from parameters to full displacement:

- `make_energy(p)` creates the **total** potential for those parameters.
- `partial(p, u)` returns sparse $B$ at the supplied fixed displacement,
  including parameter-dependent loads if the model has them.
- Every evaluation starts from the same `initial` vector. Fixed values come
  from that vector. Value-only requests skip the sensitivity solve.
- Jacobian requests use the exact, unshifted $K_{ff}$, factorized once for all
  parameter columns. Projected Hessians used by forward Newton do not replace it.

```python
from litepgo import solver

equilibrium = solver.StaticEquilibriumResponse(
    num_parameters, fixed, initial, options, make_energy, partial)
u, du_dp = equilibrium.eval(p, need_jacobian=True)
```

This requires a converged isolated differentiable equilibrium and a nonsingular
free stiffness. Forward nonconvergence or sensitivity-solve failure throws;
no approximate derivative is returned. Branch switches, singular states, and
parameter-dependent prescribed boundary data are outside this contract.
Convergence does not certify mechanical stability; the sensitivity solve accepts
indefinite nonsingular stiffness. Callbacks and captured data must be deterministic.

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
