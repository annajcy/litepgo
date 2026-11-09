# Newton method

Newton minimizes a scalar Energy while preserving prescribed DOF values. Set a
finite nonnegative gradient tolerance and start from a feasible initial state.

## Constrained Newton

Let $P$ inject the free coordinates, $H=\nabla^2E$, $g_f=P^T\nabla E$, and
$H_{ff}=P^THP$. A Newton iteration computes a descent direction from

$$
(H_{ff}+\tau I)\Delta q=-g_f,\qquad \Delta x=P\Delta q.
$$

Diagonal shifts $\tau$ are increased when factorization fails or the direction
is not descent. With `max_coordinate_step` equal to $\ell$, the line search starts at

$$
\beta=\min(1,\ell/\|\Delta x\|_\infty),\qquad
\alpha=\beta\,\texttt{energy.max\_feasible\_step}(x,\beta\Delta x).
$$

The coordinate limit therefore applies before the feasible-step query, including
CCD broad-phase construction. Infinity gives $\beta=1$. The search shrinks
$\alpha$ until the energy decreases, allowing
floating-point comparison noise. Domain errors from an evaluation propagate
immediately; feasible-step bounds must therefore match the energy's domain.

Before any evaluation or convergence return, Newton queries
`energy.max_feasible_step(initial, zeros)` to check initial feasibility. This
also applies when every DOF is fixed. The query delegates geometry validation
to the Energy implementation. Native IPC checks surface intersections and
minimum separation even for a zero direction.

```python
from litepgo import solver
options = solver.NewtonOptions(gradient_tolerance=1e-8, max_iterations=100)
result = solver.newton_solve(potential, fixed, initial, options)
```

Here `potential` is an Energy, `fixed` a DofReduction with matching dimension,
and `initial` a full vector supplying the fixed values. The
[quickstart](../quickstart.md) is a complete executable setup. Python can pass a
fifth `solver=` argument; C++ requires a `SparseSolver&`. An optional iteration
callback receives energy, free-gradient norm, pattern changes, shifts, feasibility
bound, accepted step, and line-search count. Terminal callbacks have step zero.

| Option | Default | Meaning |
| --- | --- | --- |
| `gradient_tolerance` | Required | Stop when $\|g_f\|_\infty$ is at most this value |
| `max_iterations` | 50 | Maximum accepted iterations |
| `hessian` | `Projected` | Hessian used for directions |
| `max_line_search_steps` | 30 | Trial-energy budget per direction |
| `line_search_shrink` | 0.5 | Multiplicative step reduction |
| `max_coordinate_step` | `inf` | Positive upper bound on each solver coordinate's absolute change per accepted iteration |

For `max_coordinate_step` equal to $\ell$, accepted iterates satisfy

$$
\|x_{k+1}-x_k\|_\infty
=\max_i |x_{k+1,i}-x_{k,i}|\leq\ell.
$$

The solver scales the entire direction by one common factor. For example,
with $\ell=0.01$, a proposed direction $(0.08,-0.04)$ is scaled to
$(0.01,-0.005)$ before CCD. Feasibility checks and line search can reduce the
movement further. Repeated iterations accumulate movement toward equilibrium.

`max_coordinate_step` uses the units of `initial`. For a mapped energy
$E(Aq+b)$, it bounds movement in $q$. A physical displacement bound requires
accounting for $A$. `NewtonStep.feasible_step` and `step` include the coordinate
limit and remain fractions of the original Newton direction.

Inspect `result.status`: `Converged`, `MaxIterations`, `StepTooSmall`, or
`NoDescentDirection`. `result.x` is full-sized; `energy` and `gradient_norm`
describe the returned state. A small free gradient is a stationarity criterion,
not proof of a global minimum. Rigid modes must be constrained or penalized.

Newton requests value, gradient and Hessian in one Energy evaluation. It compares
returned full Hessian structures, rebuilding the reduced assembly and symbolic
analysis when they change. `pattern_changed` is true on the first evaluation.
The factorization is reused symbolically when only numerical values change.
