# Batched local assembly

Mesh bases gather local inputs, evaluate equal-sized batches, and scatter
contributions to the global energy or response. They add no material law or
integration measure. Choose the indexing through a [DOF map](../dof/maps.md),
and follow the module's [error and feasibility rules](index.md#errors-and-feasibility).

## Batch local terms

`MeshEnergy(map, batch_size=256)` groups equal-sized local
DOF vectors. A Python `eval_batch` override receives `ids`, `inputs`, `need_gradient`, and
`need_hessian`, where `ids` are element IDs and each input row follows the map.
It returns one **integrated** energy per row and optional derivative arrays:

| Output | Shape for batch size $B$, local input count $n$ |
| --- | --- |
| Inputs / gradients | `(B, n)` |
| Values | `(B,)` |
| Hessians | `(B*n, n)`, one stacked $n\times n$ block per row |
| Feasible steps | `(B,)`, from `eval_feasible_step_batch(ids, inputs, deltas)` |

`batch_size=1` retains the batch dimension. C++ uses row-major `BatchMatrix`
and nullable output pointers instead of flags. Every requested entry must be
written, including zeros. `MeshEnergy` projects each local Hessian when requested,
then assembles the contributions.

`MeshResponse(input_map, output_map, batch_size=256)` similarly sums
local output vectors into output-map DOFs. Python returns values `(B, no)` or
`(values, Jacobians)`, with Jacobians `(B*no, ni)`, where `ni` and `no` are
the local input and output DOF counts. The two maps need matching
element order/counts. No quadrature weights are added by these generic bases.

Hooks may execute concurrently. Keep model data read-only, return finite data,
and do not rely on execution order. Assembly uses a fixed contribution order;
external linear solvers can still introduce platform-dependent rounding.

## Subclass hooks

`MeshEnergy(map, batch_size=256)` and
`MeshResponse(input_map, output_map, batch_size=256)` are subclass bases.
Override `eval_batch`; the base owns batching, validation, and global assembly.
Store model data on the subclass and implement the batch hooks in C++ or Python.

A mesh response's `eval_batch(ids, x, need_jacobian)` returns `(B,no)` values,
or `(values, Jacobians)` with Jacobians shaped `(B*no,ni)` when requested.
Optional overrides `eval_jvp_batch(ids, x, dx)` and
`eval_vjp_batch(ids, x, gy)` return `(B,no)` and `(B,ni)` respectively.
Here `x` and `dx` have shape `(B,ni)`, and `gy` has shape `(B,no)`.
Defaults evaluate local explicit Jacobians and multiply them. Python overrides may
call `super().eval_jvp_batch(ids, x, dx)` or
`super().eval_vjp_batch(ids, x, gy)` to use this fallback. Direct overrides
avoid computing Jacobian values. JVP scatters through the output map; VJP
scatters through the input map. Shared and repeated indices accumulate.
Hooks and overrides may run concurrently and must not mutate shared state.

A mesh energy's `eval_batch(ids, x, need_gradient, need_hessian)` returns
`(values, gradients, Hessians)` with shapes `(B,)`, `(B,n)`, `(B*n,n)`;
unrequested derivatives may be `None`. Override
`eval_feasible_step_batch(ids, x, dx)` for restricted domains, returning `(B,)`
safe bounds in `[0,1]`. Its default of one assumes every finite segment is feasible;
Python overrides can call `super().eval_feasible_step_batch(ids, x, dx)`.

## Jacobian products

Every `Response` supports `evaluate(x).jvp(dx)` = `J(x) @ dx` and
`evaluate(x).vjp(gy)` = `J(x).T @ gy`. Inputs and results are finite dense vectors.
The base default evaluates the explicit Jacobian on each call. C++ subclasses
and Python `FixedPatternResponse`/`DynamicPatternResponse` subclasses can override
`jvp_at(x, dx)` and `vjp_at(x, gy)` to compute products directly.
Python overrides can call `super().jvp_at(x, dx)` or
`super().vjp_at(x, gy)` to fall back to the explicit Jacobian.
Use the evaluation `jvp` / `vjp` entry points for evaluation; they retain
validation and any implementation-specific workspace protection. Hooks are subclass implementation
methods, not independent evaluation entry points.
Ordinary products prepare required patterns automatically when using the explicit Jacobian fallback.

```python
y = R.evaluate(x).value
J = R.evaluate(x).jacobian()
dy = R.evaluate(x).jvp(dx)
gx = R.evaluate(x).vjp(gy)
```

Fixed sparse subclasses retain their declared patterns even when direct product
overrides avoid explicit Jacobians; dynamic subclasses query patterns only when needed. See [Response evaluation](responses.md) for
ordinary evaluation and subclass contracts.

The ARAP and Laplacian examples define named `MeshEnergy` subclasses in both
C++ and Python, with model state stored on the object and `eval_batch` overridden.
The standalone [mesh energy](../../examples/energy/mesh-energy-subclass.md) and
[mesh response](../../examples/energy/mesh-response-subclass.md) examples show
subclass construction, batch formulas, and direct product overrides.
