# Gauss–Newton least squares

Gauss–Newton fits a residual Response targeting zero. Supply analytic Jacobians
and a finite positive gradient tolerance. A measurement becomes a residual by
subtracting its observation.

## Damped Gauss–Newton

For a residual $r(p)$ and Jacobian $J=\partial r/\partial p$, minimize

$$
\Phi(p)=\tfrac12\|r(p)\|^2,\qquad
(J^TJ+\eta I)\Delta p=-J^Tr.
$$

The solver adapts damping $\eta$, requesting only residual values for trial
points. Symbolic factorization is reused while the normal-matrix pattern remains
unchanged. It does not compute finite-difference Jacobians or impose general
parameter bounds.

```python
from litepgo import solver

options = solver.GaussNewtonOptions(gradient_tolerance=1e-10)
result = solver.gauss_newton_solve(residual, initial_parameters, options)
```

Defaults are `max_iterations=50`, `max_trial_steps=20`, and
`initial_damping=1e-3`. Statuses are `Converged`, `MaxIterations`, and
`StepTooSmall`. The result records `x`, `objective`, `gradient_norm`, accepted
iterations, and evaluations. Convergence means $\|J^Tr\|_\infty$ is small;
it does not require zero residual. Normal equations square the spectral condition
number before damping, so scale parameters and observations appropriately.
The [curve-fit example](../../examples/solvers/gauss-newton-fit.md) supplies
a complete residual implementation.
