# Batched local assembly

Mesh wrappers gather local inputs, evaluate equal-sized batches, and scatter
contributions to the global energy or response. They add no material law or
integration measure. Choose the indexing through a [DOF map](../dof/maps.md),
and follow the module's [error and feasibility rules](index.md#errors-and-feasibility).

## Batch local terms

`CustomMeshEnergy(map, eval_batch, batch_size=256, step=None)` groups equal-sized local
DOF vectors. A Python callback receives `ids`, `inputs`, `need_gradient`, and
`need_hessian`, where `ids` are element IDs and each input row follows the map.
It returns one **integrated** energy per row and optional derivative arrays:

| Output | Shape for batch size $B$, local input count $n$ |
| --- | --- |
| Inputs / gradients | `(B, n)` |
| Values | `(B,)` |
| Hessians | `(B*n, n)`, one stacked $n\times n$ block per row |
| Feasible steps | `(B,)`, from `step(ids, inputs, deltas)` |

`batch_size=1` retains the batch dimension. C++ uses row-major `BatchMatrix`
and nullable output pointers instead of flags. Every requested entry must be
written, including zeros. `CustomMeshEnergy` projects each local Hessian when requested,
then assembles the contributions.

`CustomMeshResponse(input_map, output_map, eval_batch, batch_size=256)` similarly sums
local output vectors into output-map DOFs. Python returns values `(B, no)` or
`(values, Jacobians)`, with Jacobians `(B*no, ni)`, where `ni` and `no` are
the local input and output DOF counts. The two maps need matching
element order/counts. No quadrature weights are added by these generic wrappers.

Callbacks may execute concurrently. Keep captures read-only, return finite data,
and do not rely on execution order. Assembly uses a fixed contribution order;
external linear solvers can still introduce platform-dependent rounding.

## Subclasses and callback adapters

`MeshEnergy(map, batch_size=256)` and
`MeshResponse(input_map, output_map, batch_size=256)` are subclass bases.
Override `eval_batch`; the base owns batching, validation, and global assembly.
For existing callback-based code, replace the constructor name with
`CustomMeshEnergy` or `CustomMeshResponse`; callback signatures are unchanged.
These adapter classes are also available in C++ in the mesh energy/response headers.

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
Callbacks and overrides may run concurrently and must not mutate shared state.

A mesh energy's `eval_batch(ids, x, need_gradient, need_hessian)` returns
`(values, gradients, Hessians)` with shapes `(B,)`, `(B,n)`, `(B*n,n)`;
unrequested derivatives may be `None`. Override
`eval_feasible_step_batch(ids, x, dx)` for restricted domains, returning `(B,)`
safe bounds in `[0,1]`. Its default of one assumes every finite segment is feasible;
Python overrides can call `super().eval_feasible_step_batch(ids, x, dx)`.

## Jacobian products

Every `Response` supports `eval_jvp(x, dx)` = `J(x) @ dx` and
`eval_vjp(x, gy)` = `J(x).T @ gy`. Inputs and results are finite dense vectors.
The base default evaluates the explicit Jacobian on each call. C++ subclasses
and Python `FixedPatternResponse`/`DynamicPatternResponse` subclasses can override
`eval_jvp_values(x, dx)` and `eval_vjp_values(x, gy)` to compute products directly.
Python overrides can call `super().eval_jvp_values(x, dx)` or
`super().eval_vjp_values(x, gy)` to fall back to the explicit Jacobian.
Use the public `eval_jvp` / `eval_vjp` entry points for evaluation; they retain
validation and the one-call-at-a-time guard. Hooks are subclass implementation
methods, not independent evaluation entry points.
For dynamic patterns, call `update_pattern(x)` first, just as for explicit Jacobians.

```python
R.update_pattern(x)
y = R.eval_value(x)
J = R.eval_jacobian(x)
dy = R.eval_jvp(x, dx)
gx = R.eval_vjp(x, gy)
```

Explicit Jacobians and Gauss–Newton retain their existing behavior. This change
does not make Jacobian patterns lazy: objects still store their declared pattern,
even when direct product overrides avoid evaluating its numerical entries.

The ARAP and Laplacian examples define named `MeshEnergy` subclasses in both
C++ and Python, with model state stored on the object and `eval_batch` overridden.
Use `CustomMeshEnergy`/`CustomMeshResponse` for injected formulas and short-lived
callback adapters, including the test-only formula oracles.
