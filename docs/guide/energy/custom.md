# Custom energies and responses

Implement a scalar $E:\mathbb R^n\to\mathbb R$ or a vector $r:\mathbb R^n\to\mathbb R^m$
through fixed/dynamic-pattern subclasses. Read [evaluation contracts](evaluation.md)
and the module's [error and feasibility rules](index.md#errors-and-feasibility)
before overriding the numerical hooks.

## Implement a scalar energy

Derive from `FixedPatternEnergy(pattern)` or
`DynamicPatternEnergy(num_dofs)`. The latter additionally
implements `hessian_pattern_at(x)`. In Python, override

```python
def eval_values(self, x, want_gradient, hessian_kind):
    # Return (scalar, gradient_or_None, hessian_or_None).
    # hessian_kind is None, Exact, or Projected.
    ...
```

Return `None` for an infeasible state. Return a dense gradient and a dense or SciPy sparse Hessian
with correct dimensions and no nonzero entries outside the declared structure.
Implement both requested Hessian kinds; no automatic projection is added to a
custom scalar subclass. Optional `feasible_step(x, dx)` defaults to one, which
is appropriate only if the complete finite segment is feasible.

C++ overrides `double eval_values(x, gradient*, hessian*, kind) const` and, for a
dynamic energy, `SpMat hessian_pattern_at(x) const`. Null derivative pointers
mean that output was not requested; `kind` matters only when the Hessian pointer
is non-null. Overwrite every requested gradient and stored Hessian entry,
including zeros, without changing the prepared Hessian structure. Subclass hooks
belong to the implementation; callers use the validated public API. The
[custom-energy example](../../examples/energy/custom-energy.md) includes
complete implementations in both languages.

For implementations that naturally produce their own sparse matrices, derive
from `Energy(num_dofs)` directly in C++ and override
`eval_impl(x, need_gradient, optional_hessian_kind) const`, returning an
`EnergyEvaluation` with exactly the requested outputs. `feasible_step` is shared
by both construction paths. The internal `detail::SparseEnergy` adapter supplies
prepared local buffers for Fixed/Dynamic subclasses; callers never use those
buffers. `MeshEnergy` retains its batch hooks and protects its shared workspace.

## Implement a vector response

`FixedPatternResponse(pattern)` gets $m,n$ from the pattern's rows and columns.
`DynamicPatternResponse(num_inputs, num_outputs)` additionally implements
`jacobian_pattern_at(x)`. Override `eval_impl(x, need_jacobian)` and return the
value for a value-only call or `(value, J)` for a Jacobian request. Return `None`
for an infeasible point. Native `eval_impl(x, SpMat* J)` uses a nullable
Jacobian pointer and overwrites its prepared numerical entries.

```python
value = response.eval_value(x)
value, J = response.eval_value_and_jacobian(x)  # J has shape (m, n)
```

Ordinary calls prepare their required structure in local storage. No explicit
pattern preparation or output buffer protocol is exposed. See
[Response evaluation](responses.md) for all public methods, subclass hooks, and
migration from the previous API. The
[response example](../../examples/energy/custom-response.md) demonstrates fixed
and dynamic structure and a Gauss–Newton solve.

Optional second derivatives use an additional Hessian pattern and the
`eval_hessian_impl` hook. See [Response Hessians](response-hessians.md) for the
weighted definition and constructor signatures. First-order subclasses need no
Hessian hook, but must use the current evaluation hook names.
