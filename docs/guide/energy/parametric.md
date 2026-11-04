# Parameterized energies

`energy.ParametricEnergy` defines a total potential `E(u, p)`, including
parameter-dependent conservative loads. It is separate from the single-input
`Energy` interface. The native base and Python subclass protocol share the same
two views:

- `model.energy_at(p)` returns an `Energy` in displacement `u`.
- `model.force_at(u)` returns a `Response` in parameters `p`, with value
  `g = grad_u E`, Jacobian `B = dg/dp`, and products `B v` / `B^T w`.

Both views obtain `g` through the same state evaluation hook. There is no
independently supplied force-value callback. Analytic derivative hooks must still
be mathematically consistent; the interface cannot verify arbitrary handwritten
formulas.

## Implement a model

Python subclasses initialize
`ParametricEnergy(state_pattern, parameter_pattern, fixed_pattern=True)`.
The state pattern is `n x n` for the displacement Hessian; the parameter pattern
is `n x m` for the force parameter Jacobian. Dimensions are inferred, exposed as
`num_dofs` and `num_parameters`, and never change. Patterns are canonical sparse
structures; Python also accepts dense matrices.

```python
import numpy as np
from litepgo import energy, solver

class QuadraticModel(energy.ParametricEnergy):
    def __init__(self):
        super().__init__(np.eye(2), np.eye(2))

    def eval_state_values(self, u, p, need_gradient, hessian_kind):
        return (
            0.5 * u @ u - p @ u,
            u - p if need_gradient else None,
            np.eye(2) if hessian_kind is not None else None,
        )

    def eval_parameter_jacobian(self, u, p):
        return -np.eye(2)

model = QuadraticModel()
potential = model.energy_at(np.array([2., 3.]))
force = model.force_at(np.zeros(2))
np.testing.assert_allclose(potential.eval_gradient(np.zeros(2)),
                           force.eval_value(np.array([2., 3.])))
```

`eval_state_values` returns `(value, gradient, Hessian)`; unrequested derivatives
may be `None`, and returning `None` for the whole result indicates infeasibility.
The force view requests the gradient from this same hook. `hessian_kind` is
`None`, `Exact`, or `Projected`. The view forwards projection requests unchanged:
a mesh implementation must preserve its per-element projection semantics.

Optional overrides `eval_parameter_jvp(u, p, v)` and
`eval_parameter_vjp(u, p, w)` avoid explicit `B`. Their defaults evaluate `B`
and multiply; Python overrides may call `super()` for this fallback.
`eval_feasible_step(u, p, du)` returns a bound in `[0,1]`, defaulting to one;
restricted domains must override it.

For dynamic patterns, set `fixed_pattern=False` and override
`state_pattern_at(u, p)` / `parameter_pattern_at(u, p)` to return complete
patterns for the supplied pair. The defaults return the initial patterns.
Each view owns its current pattern; updating one view never updates another.
Call `update_pattern` before requesting derivatives as with existing
Energy/Response objects.

C++ derives from `energy::ParametricEnergy` in
`litepgo/energy/parametric/parametric_energy.hpp`. Its constructor takes the same
two sparse patterns and flag. The protected hooks use `ConstVecRef` inputs:
`eval_state_values(u,p,Vec* g,SpMat* K,HessianKind)` returns the energy, and
`eval_parameter_jacobian(u,p,SpMat& B)` fills only values within the provided
pattern. Product hooks return `Vec`; pattern hooks return `SpMat`.

Construct C++ models with `std::make_shared<Model>(...)` before calling view
factories. Views retain model ownership and copy the fixed argument. Python
views keep subclass overrides alive; ownership cycles with models are collectable.
Configuration must remain fixed, and concurrent/reentrant calls through any views
of the same model are rejected. The base does not cache evaluations or
factorizations. Jacobian patterns are still stored explicitly.

## Equilibrium and migration

```python
equilibrium = solver.StaticEquilibriumResponse(model, fixed, initial, options)
```

C++ accepts the same arguments, with `std::shared_ptr<ParametricEnergy>` first.
The old separate energy-factory/partial constructor is removed. Move existing
state evaluation, exact parameter partials, and feasibility rules into one model.
Existing FEM energy and force classes remain usable inside analytic model hooks;
fixed geometry and material/plastic fields retain their existing field-list
conventions.

Equilibrium uses the model's Energy view for Newton and its force view for
parameter derivatives. Explicit sensitivity requests assemble `B`; JVP/VJP
delegate to the force products and can avoid it entirely. See
[equilibrium sensitivities](../solver/equilibrium.md).

This interface does not supply an automatic-differentiation backend. The Torch
material implementation and a general parametric mesh assembler are not included;
the existing Torch response bridge can consume the resulting equilibrium response.
