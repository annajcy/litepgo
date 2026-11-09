# Response evaluation

A Response is a fixed-configured map $y=R(x)$ from $n$ input scalars to $m$
output scalars. Only the explicit input is differentiated. Configuration such as
an affine matrix, topology, or material model stays fixed. Results must not depend
on the order of previous evaluations.

## Public API

Create `e = r.evaluate(x)` once and reuse it:

| Python call | Mathematical result | Shape |
| --- | --- | --- |
| `e.value` | $R(x)$ | `(m,)` |
| `e.jacobian()` | $J_R(x)$ | `(m,n)` |
| `e.jvp(dx)` | $J_R(x)dx$ | `(m,)` |
| `e.vjp(weights)` | $J_R(x)^T weights$ | `(n,)` |
| `e.hessian(weights)` | $\sum_i weights_i\nabla_x^2R_i(x)$ | `(n,n)` |

C++ uses `auto e = r.evaluate(x)` followed by `e->value()`, `e->jacobian()`,
`e->jvp(dx)`, `e->vjp(weights)` and `e->hessian(weights)`. The old Response
`eval_*` entry points have been removed; Energy APIs are unchanged.
For joint value/Jacobian work, request the Jacobian first, then read the value.
The default sparse implementation saves the value produced with the Jacobian.

Evaluations copy the input and prepare results on demand. Merely creating one
validates the input; infeasible-point errors can occur on first value/derivative use.
An evaluation retains its value. Equilibrium evaluations additionally retain their
solved state and lazily prepare exact stiffness for reuse. Independent evaluations
never share a last-input cache. Model and callback configuration must remain fixed
while an evaluation is in use. Use each evaluation from one caller at a time.
Shared C++ responses are retained automatically; a stack-allocated response must
outlive its evaluations. Python keeps the response alive and returns value copies.
C++ `value()` returns a const reference valid for the evaluation's lifetime.

Matrices are SciPy CSC matrices in Python and `SpMat` in C++. Vectors are NumPy
arrays / `Vec`. Returned matrices own their storage. Output weights are fixed during
differentiation; Hessians are full symmetric matrices. Unsupported second derivatives
raise an error, even for zero weights. Zero curvature is supported second differentiation.

`num_inputs`, `num_outputs`, and `has_hessian` are read-only Python properties and
C++ queries. Dimensions and capabilities stay fixed for an object's lifetime.
Numerical evaluation methods are non-const in C++ to allow implementation workspaces.

The mathematical `Response` base owns no pattern cache or concurrency flag. Its
public entry points validate inputs and returned results. Direct C++ implementations
return owning results from `value_at`, `value_jacobian_at`, and
optional `hessian_at`. Products default to explicit Jacobian
multiplication; direct `jvp_at` / `vjp_at` overrides avoid that work.

## Sparse subclasses

`FixedPatternResponse(pattern, hessian_pattern=None)` owns fixed structure snapshots.
An omitted Hessian pattern means unsupported; an empty square pattern declares zero
curvature. Stored numerical values in patterns are ignored, including explicit zeros.

`DynamicPatternResponse(num_inputs, num_outputs)` declares dimensions without an
initial pattern. It implements `jacobian_pattern_at(x)` and, when supporting second
order, `hessian_pattern_at(x)`. Each evaluation queries only the structure it needs;
no previous input or current pattern is retained. Hessian structure is the union of
component Hessian structures and is independent of weights. Conservative supersets
are allowed, and pattern changes do not establish differentiability at switching points.

Both sparse bases use the same numerical hooks:

```cpp
Vec eval_impl(const ConstVecRef& x, SpMat* jacobian) override;
Vec jvp_at(const ConstVecRef& x, const ConstVecRef& dx) override;
Vec vjp_at(const ConstVecRef& x, const ConstVecRef& weights) override;
void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) override;
```

A null Jacobian requests only the value. Otherwise, the implementation receives a
zero-valued matrix with prepared structure and overwrites all stored entries. It
must not change dimensions or structure. Allocation and debug structural checks are shared by an internal adapter,
not exposed as another public extension API.

Python `eval_impl(self, x, need_jacobian)` returns the value or `(value, Jacobian)`.
`eval_hessian_impl(self, x, weights)` returns a matrix. Return `None` for an infeasible
point. Product overrides may use `super().jvp_at` or `super().vjp_at`
for the explicit Jacobian fallback. Python numerical hooks may make finite reentrant
calls through the public evaluation methods; nested calls still dispatch to the
Python override. Calling `super()` explicitly selects the product fallback.

A dynamic subclass declares second-order support by overriding C++ `has_hessian()`
or the Python property:

```python
@property
def has_hessian(self):
    return True
```

This declaration requires both Hessian hooks. Missing implementations raise an
error; they never silently supply zero curvature. C++ `has_hessian()` is not
`noexcept`, so exceptions from Python capability queries can propagate normally.

## Solvers and composition

Gauss–Newton uses joint value/Jacobian evaluation at each iterate and value-only
evaluation at trial points. It compares the structure of the normal matrix before
symbolic analysis. It does not manage Response patterns or output buffers.

Composed responses use independent results from child evaluations, including when
both roles share a child. They support second order when both children support it.
See [Response Hessians](response-hessians.md) for the chain rule.

Elementwise evaluations retain one child evaluation per element. Reading values
and then requesting products or a Jacobian reuses those child states, including
when the transform is an equilibrium response. Child calls remain sequential
because the transform may have shared workspaces.

## Ownership

Composed and elementwise responses retain their children. Children and callbacks
must not strongly reference their owner; use a weak back-reference when necessary.
Bindings keep children alive but do not provide cross-language cycle collection.
See the [Python ownership contract](../concepts.md#python-ownership-must-remain-acyclic).

## Concurrency

The base does not prohibit concurrent calls or reentry. This does not make arbitrary
subclasses thread-safe: subclasses must protect any shared mutable workspace.
`MeshResponse` rejects overlapping numerical calls on the same instance because it
reuses batch workspaces. Its guard is released on exceptions. Internal parallel
batch execution remains supported; batch hooks must obey their thread-safety contract.
Read-only affine and exponential responses use per-call results without this guard.

## Validation in Debug and Release

Every build checks dimensions, second-order capability, and finite input/output
values. Fixed patterns are normalized and validated once at construction; dynamic
patterns are normalized and validated once when generated for an evaluation.
Python derivative conversion always checks shape, finite values, and nonzero
entries outside the declared structure.

When `NDEBUG` is not defined (normally Debug), native evaluation also verifies
that filling a sparse matrix did not change its structure, that returned matrices
are canonical, and that Hessians have symmetric structure and values. Violations
raise errors. These are implementation-contract checks: Release (`NDEBUG` defined)
trusts C++ implementations to preserve structure and return symmetric Hessians.
The contracts still apply, but violations are not guaranteed to be diagnosed.
Release does not retain a comparison-only pattern copy or construct a transpose
solely to check Hessian symmetry. Numerical finite-difference and symmetry tests
continue to verify valid implementations in both configurations.

## Migration

Replace Response `eval_value(x)` with `evaluate(x).value` in Python or
`evaluate(x)->value()` in C++. Move Jacobian, Hessian and product calls onto the
same evaluation object and omit the input argument there. Replace joint calls
with `J = e.jacobian()` followed by `value = e.value`. Reusing `e` is particularly
important for equilibrium solves. Existing `eval_value_and_jacobian` and other
public Response `eval_*` methods are no longer available.

Direct C++ response kernels now use `value_at`, `value_jacobian_at`, `hessian_at`,
`jvp_at` and `vjp_at`. Python sparse subclasses use `jvp_at`/`vjp_at` for custom
products; sparse `eval_impl`, `eval_hessian_impl`, and mesh batch hooks retain
their numerical-kernel roles. Custom C++ stateful implementations can override
`make_evaluation` and return a `ResponseEvaluation` with per-input storage.

The public `has_fixed_pattern`, `update_patterns`, `jacobian_pattern`,
`hessian_pattern`, and native `_into` interfaces are removed. Call ordinary evaluation
and inspect the returned matrix when structure is needed. Dynamic constructors take
dimensions, and dynamic Hessian support is declared through `has_hessian`, not a
constructor flag or initial Hessian pattern. Sparse numerical hooks are non-const;
structure hooks remain const. Energy has its own scalar evaluation API. FEM force maps use this same Response contract.

Returned matrices may allocate. This API does not promise allocation-free evaluation
or expose a symbolic preparation protocol.
