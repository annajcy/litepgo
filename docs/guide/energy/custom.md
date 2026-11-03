# Custom energies and responses

Implement a scalar $E:\mathbb R^n\to\mathbb R$ or a vector $r:\mathbb R^n\to\mathbb R^m$
through fixed/dynamic-pattern subclasses. Read [evaluation contracts](evaluation.md)
and the module's [error and feasibility rules](index.md#errors-and-feasibility)
before overriding the numerical hooks.

## Implement a scalar energy

Derive from `FixedPatternEnergy(num_dofs, pattern)` or
`DynamicPatternEnergy(num_dofs, initial_pattern)`. The latter additionally
implements `pattern_at(x)`. In Python, override

```python
def eval_values(self, x, want_gradient, hessian_kind):
    # Return (scalar, gradient_or_None, hessian_or_None).
    # hessian_kind is None, Exact, or Projected.
    ...
```

Return `None` for an infeasible state. Return dense or SciPy sparse derivatives
with correct dimensions and no nonzero entries outside the declared structure.
Implement both requested Hessian kinds; no automatic projection is added to a
custom scalar subclass. Optional `feasible_step(x, dx)` defaults to one, which
is appropriate only if the complete finite segment is feasible.

C++ overrides `eval_values(x, gradient*, hessian*, kind)` and, for a dynamic
energy, `refresh_pattern(x)` returning an optional sparse pattern. Private hooks
belong to the derived class; callers use the validated public API. The
[custom-energy example](../../examples/energy/custom-energy.md) includes
complete implementations in both languages.

## Implement a vector response

`FixedPatternResponse(pattern)` gets $m,n$ from the pattern's rows and columns.
`DynamicPatternResponse(initial_pattern)` adds `pattern_at(x)` in Python.
Override `eval_values(x, need_jacobian)` and return `r` for a value-only call or
`(r, J)` when a Jacobian is requested. Return `None` for an infeasible point.

```python
r, J = response.eval(x, need_jacobian=True)  # J has shape (m, n)
```

For standalone derivatives, refresh the pattern first. A dynamic response's
rows/columns remain fixed while structural entries may change. The
[response example](../../examples/energy/custom-response.md) demonstrates
both branches and a solve. C++ writes into the supplied sparse output without
changing its pattern.
