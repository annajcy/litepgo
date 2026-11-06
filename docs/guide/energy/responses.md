# Response evaluation

A Response is a fixed-configured map $y=R(x)$ from $n$ input scalars to $m$
output scalars. Only the explicit input is differentiated. Configuration such as
an affine matrix, topology, or material model stays fixed. Internal sparse
patterns and scratch storage may change; no previous input is an implicit argument.

## Ordinary evaluation

| Python call | Mathematical result | Shape |
| --- | --- | --- |
| `r.eval_value(x)` | $R(x)$ | `(m,)` |
| `r.eval_jacobian(x)` | $J_R(x)$ | `(m,n)` |
| `r.eval_value_and_jacobian(x)` | $(R(x),J_R(x))$ | tuple of the above |
| `r.eval_jvp(x, dx)` | $J_R(x)dx$ | `(m,)` |
| `r.eval_vjp(x, weights)` | $J_R(x)^T weights$ | `(n,)` |
| `r.eval_hessian(x, weights)` | $\nabla_x^2(weights^TR(x))$ | `(n,n)` |

Matrices are SciPy CSC matrices in Python and `SpMat` in C++. Vectors are NumPy
arrays / `Vec`. Output weights are fixed during differentiation. The Hessian is
the full symmetric matrix; see [Response Hessians](response-hessians.md).

Every ordinary call evaluates at its own input and automatically prepares the
structure it needs. You do not need to call `update_patterns` first. Value-only
calls do not prepare derivative patterns. The base prepares only the requested derivative pattern. A composed response may
also prepare its children’s supported patterns to determine the composed structure.
Direct JVP/VJP implementations do not need pattern preparation; their default
implementations prepare and multiply the explicit Jacobian.

```python
import numpy as np
from litepgo import energy

r = energy.ComposedResponse(energy.ExpResponse(2), energy.AffineMappedResponse(np.eye(2)))
x = np.array([0.1, -0.2])
y, J = r.eval_value_and_jacobian(x)
g = r.eval_vjp(x, weights=np.ones(2))
H = r.eval_hessian(x, weights=np.ones(2))
np.testing.assert_allclose(g, y)
np.testing.assert_allclose(H.toarray(), np.diag(y))
```

`num_inputs` and `num_outputs` are fixed dimensions. `has_hessian` indicates
second-order support; unsupported requests raise an error, even for zero weights.
`has_fixed_pattern` describes both supported derivative structures. In C++, these
queries are `const noexcept` methods, and ordinary evaluation methods are
non-const because they can update execution state. `eval_value_and_jacobian`
returns `std::pair<Vec, SpMat>`. Returned results own their storage.

## Explicit native buffer reuse

Solvers can prepare patterns explicitly, detect structural changes, and reuse
buffers and symbolic factorizations:

```cpp
response.update_patterns(x);
litepgo::SpMat J = response.jacobian_pattern();
litepgo::Vec y = response.eval_value_and_jacobian_into(x, J);

if (response.update_patterns(next_x))
    J = response.jacobian_pattern();
response.eval_jacobian_into(next_x, J);
```

`update_patterns(x)` updates both supported patterns and returns true if either
changes. It stores no input or numerical derivative cache. Pattern accessors
contain zero values. Python exposes copies of these patterns and the explicit
update method, but does not expose the native `_into` methods.

`eval_jacobian_into(x, J)` and `eval_hessian_into(x, weights, H)` return void.
`eval_value_and_jacobian_into(x, J)` returns the value while writing the Jacobian.
They overwrite every stored matrix value, without resizing or accumulating.
Buffers must match the prepared pattern, and that pattern must describe the
current input; a stale pattern is rejected before numerical writes. Pattern
hooks may be consulted for validation, but these calls do not install new
patterns on the evaluated Response. Shared children may prepare their own
patterns during composition.

Re-copy buffers whenever their structure no longer matches the Response's
current pattern, including after ordinary evaluation at a different point.
Inputs/weights must not alias writable output storage. Argument and pattern validation errors leave
outputs untouched; after an exception during numerical evaluation, buffer
contents are unspecified.

## Implementing a Response

Derive from `FixedPatternResponse` or `DynamicPatternResponse`. Both use these
native numerical hooks:

```cpp
Vec eval_impl(const ConstVecRef& x, SpMat* jacobian) const override;
Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& dx) override;
Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& weights) override;
void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) const override;
```

Only `eval_impl` is mandatory: a null pointer requests value only, otherwise
write the prepared Jacobian values too. Products default to Jacobian multiplication;
their hooks are non-const so the fallback can prepare a dynamic pattern. Implement
`eval_hessian_impl` when opting into Hessian support at construction.

Python implements `eval_impl(self, x, need_jacobian)`, returning the value or
`(value, Jacobian)`, and optional `eval_jvp_impl`, `eval_vjp_impl`, and
`eval_hessian_impl`. Python Hessian hooks return a matrix. `None` denotes an
infeasible evaluation. See [custom responses](custom.md#implement-a-vector-response).

Dynamic subclasses implement `jacobian_pattern_at(x) const` in C++ or
`jacobian_pattern_at(self, x)` in Python. Optional `hessian_pattern_at` defaults
to retaining the current Hessian structure. Hooks describe structural supersets;
values are ignored. They do not install patterns themselves. There is no separate
`refresh_*` hook layer. Public methods validate arguments/results and reject
concurrent or reentrant calls on the same instance. Do not call the same object's
public evaluation methods from its own numerical hooks; Python product overrides
can use `super().eval_jvp_impl` / `super().eval_vjp_impl` for the default fallback.

## Migration from the previous Response API

| Previous | Current |
| --- | --- |
| `eval(x)` for value | `eval_value(x)` |
| Python `eval(x, need_jacobian=True)` | `eval_value_and_jacobian(x)` |
| C++ `eval(x, &J)` | `eval_value_and_jacobian_into(x, J)` after explicit preparation |
| `update_pattern(x)` | `update_patterns(x)`; optional for ordinary evaluation |
| `eval_values` | `eval_impl` |
| `eval_jvp_values` / `eval_vjp_values` | `eval_jvp_impl` / `eval_vjp_impl` |
| `eval_hessian_values` | `eval_hessian_impl` |
| Dynamic `pattern_at` | `jacobian_pattern_at` |
| Response `refresh_pattern` / `refresh_hessian_pattern` | `jacobian_pattern_at` / `hessian_pattern_at`, returning `SpMat`, const |

The old Response names are removed. The separate Energy and FEM LocalResponse
interfaces keep their own names and contracts. Domain failures, malformed inputs,
and broken derivative implementations remain distinct errors; changing a sparse
pattern does not establish differentiability at a switching point.
