# Newton method

Newton minimizes a scalar Energy while preserving prescribed DOF values. Supply a convergence policy and start from a feasible initial state.

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
\alpha=\beta\,\texttt{evaluation.max\_feasible\_step}(\beta\Delta x).
$$

The coordinate limit therefore applies before the feasible-step query, including
CCD broad-phase construction. Infinity gives $\beta=1$. The search shrinks
$\alpha$ until the energy decreases, allowing
floating-point comparison noise. Domain errors from an evaluation propagate
immediately; feasible-step bounds must therefore match the energy's domain.

Before any evaluation or convergence return, Newton queries
`energy.evaluate(initial).max_feasible_step(zeros)` to check initial feasibility. This
also applies when every DOF is fixed. The query delegates geometry validation
to the Energy implementation. Native IPC checks surface intersections and
minimum separation even for a zero direction.

```python
from litepgo import solver
options = solver.NewtonOptions(max_iterations=100)
result = solver.newton_solve(potential, fixed, initial, options,
                             convergence=solver.AbsoluteConvergence(1e-8))
```

Here `potential` is an Energy, `fixed` a DofReduction with matching dimension,
and `initial` a full vector supplying the fixed values. The
[quickstart](../quickstart.md) is a complete executable setup. Python can pass a
fifth `solver=` argument; C++ requires a `SparseSolver&`. An optional iteration
callback receives energy, free-gradient norm, pattern changes, shifts, feasibility
bound, accepted step, and line-search count. Terminal callbacks have step zero.

| Option | Default | Meaning |
| --- | --- | --- |
| `max_iterations` | 50 | Maximum accepted iterations |
| `project_hessian` | `True` | Use the PSD approximation; `False` selects exact Hessian |
| `max_line_search_steps` | 30 | Trial-energy budget per direction |
| `line_search_shrink` | 0.5 | Multiplicative step reduction |
| `max_coordinate_step` | `inf` | Positive upper bound on each solver coordinate's absolute change per accepted iteration |
| `detect_precision_stagnation` | `False` | Enable the conservative diagnostic below; only exact, unshifted directions qualify |
| `precision_coordinate_scale` | `1.0` | Finite positive characteristic scale in solver coordinates; choose for the problem's units |

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
`NoDescentDirection`, or `PrecisionStagnation`. `result.x` is full-sized; `energy` and `gradient_norm`
describe the returned state. A small free gradient is a stationarity criterion,
not proof of a global minimum. Rigid modes must be constrained or penalized.

## Two-stage convergence policy

`convergence` is required and keyword-only in Python. It accepts the native
`AbsoluteConvergence(tolerance)` or a callable `ConvergenceState -> bool`.
The native absolute policy requires a finite nonnegative tolerance and checks
`gradient_norm <= tolerance` entirely in C++. A Python callable crosses the
language boundary at most twice per iteration. Exceptions propagate.
C++ takes a `ConvergenceCheck` after the sparse solver argument.
`NewtonOptions.gradient_tolerance` has been removed, without a compatibility alias.

The same callback is invoked at two points:

1. After gradient and energy evaluation, before Hessian assembly. The three
   direction diagnostics are `None`. Returning true avoids Hessian work.
2. If still unconverged and within the iteration budget, after computing a descent
   direction. Direction diagnostics now contain values. Returning true accepts
   the current iterate, before coordinate limiting, CCD or line search.

`state` is a read-only scalar snapshot, safe to retain. Reading its fields does
not trigger computation. The underlying EnergyEvaluation caches its requests.
Callbacks should be pure: two calls may refer to the same iteration.

| Field | Meaning |
| --- | --- |
| `iteration` | Number of accepted steps |
| `energy` | Current total energy |
| `gradient_norm` | Infinity norm of the free-coordinate gradient |
| `project_hessian` | Whether Newton uses projected curvature |
| `newton_correction_norm` | Infinity norm of the complete free-coordinate direction, before any limiting |
| `diagonal_shifts` | Number of diagonal shifts used to obtain this direction; zero means unshifted |
| `linear_backward_error` | `norm(A*d + g, inf) / (norm(A, inf)*norm(d, inf) + norm(g, inf))`, for the actual reduced, possibly shifted matrix |

The matrix norm is the maximum absolute row sum. A small backward error means
an accurate solve of the chosen matrix; it does not imply exact curvature or
positive definiteness. Check `not project_hessian` and `diagonal_shifts == 0`
when requiring an exact unshifted Newton correction.

```python
def convergence(state):
    if state.gradient_norm <= absolute_tolerance:
        return True
    if state.newton_correction_norm is None:
        return False
    return (
        state.gradient_norm <= relative_tolerance * force_scale
        and not state.project_hessian
        and state.diagonal_shifts == 0
        and state.linear_backward_error <= 1e-12
        and state.newton_correction_norm <= correction_tolerance * coordinate_scale
    )
```

Scales and tolerances are supplied by the application in solver coordinates.
For contacting structures, a fixed reference from the separate elastic and
contact force magnitudes is useful; the small initial continuation residual is
not a suitable force scale. Validate the resulting force-label accuracy.

Convergence takes precedence over termination diagnostics. At the iteration
budget only stage one runs. If a policy rejects a zero free gradient or an
all-fixed state, Newton returns `NoDescentDirection`; it does not manufacture
uncomputed direction diagnostics or force acceptance.

Newton retains the accepted EnergyEvaluation for the next iteration. Line search
requests trial values only. When a Hessian is needed, its structure is compared
to the previous one and symbolic analysis is reused when unchanged.
`pattern_changed` is true on the first Hessian assembly, not on a terminal
stage-one return that never requests a Hessian.

## Precision stagnation (opt-in)

```python
options = solver.NewtonOptions(
    max_iterations=1500,
    project_hessian=False,
    detect_precision_stagnation=True,
    precision_coordinate_scale=1.0,  # appropriate for a unit-cell fluctuation coordinate
)
```

`PrecisionStagnation` is **not convergence**: the convergence policy
did not accept the state. It diagnoses numerical stagnation; it does not prove an unattainable
tolerance, a correct derivative implementation, or a stable minimum. Callers must
not silently treat it as `Converged`. StaticEquilibriumResponse continues to reject
all nonconverged outcomes. Detection is disabled by default and inactive for
projected Hessians or diagonally shifted directions.

For the complete direction before coordinate capping, CCD, or line search, define

$$
\eta_x=\max_{i\in\mathrm{free}}\frac{|d_i|}{\max(|x_i|,s)},\qquad
\eta_L=\frac{\|H_{ff}d_f+g_f\|_\infty}
{\|H_{ff}\|_\infty\|d_f\|_\infty+\|g_f\|_\infty}.
$$

Here `s=precision_coordinate_scale`; it is a coordinate scale, not a force
scale. A mapped energy uses solver coordinates, not necessarily physical positions.
The detector requires exact unshifted curvature, `eta_x <= 128 * double epsilon`
and finite linear backward error `eta_L <= 1e-12`. It then checks three consecutive
qualifying full-step iterations with energy changes at most
`16 * epsilon * max(1, abs(current_energy))`. Any new best residual restarts the
window; a nonqualifying or shortened step clears it. Thus tiny steps that keep
improving the residual are not automatically labeled precision stagnation.

There is one immediate case: a reliable full direction with feasible step one
cannot change any coordinate (`x + d == x`). That returns `PrecisionStagnation`
without a three-step window. If only CCD, coordinate capping, or line-search
shrinking makes a step unrepresentable, the outcome remains `StepTooSmall`.
Convergence is checked first and the iteration budget retains its existing priority.

`NewtonStep` and `NewtonResult` expose `scaled_direction_norm`,
`linear_backward_error`, and `precision_stagnation_steps`. Uncomputed numeric
diagnostics are NaN. The final callback has step zero; returned x, energy and
residual describe the current evaluated state, not the uncommitted trial.
The detector does not establish positive definiteness; successful factorization
with an arbitrary SparseSolver does not prove local stability.

Tests cover deterministic roundoff oscillation, unrepresentable full corrections,
normal convergence, improving tiny corrections, coordinate limits, synthetic and
native IPC CCD restrictions, and shifted/projected directions. A unit-cell
Neovius tight-contact reproduction also terminates with this status while the
already-converged checkpoint retains `Converged`.
