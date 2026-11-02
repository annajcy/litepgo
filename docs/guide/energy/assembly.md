# Batched local assembly

Mesh wrappers gather local inputs, evaluate equal-sized batches, and scatter
contributions to the global energy or response. They add no material law or
integration measure. Choose the indexing through a [DOF map](../dof/maps.md),
and follow the module's [error and feasibility rules](index.md#errors-and-feasibility).

## Batch local terms

`MeshEnergy(map, eval_batch, batch_size=256, step=None)` groups equal-sized local
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
written, including zeros. `MeshEnergy` projects each local Hessian when requested,
then assembles the contributions.

`MeshResponse(input_map, output_map, eval_batch, batch_size=256)` similarly sums
local output vectors into output-map DOFs. Python returns values `(B, no)` or
`(values, Jacobians)`, with Jacobians `(B*no, ni)`, where `ni` and `no` are
the local input and output DOF counts. The two maps need matching
element order/counts. No quadrature weights are added by these generic wrappers.

Callbacks may execute concurrently. Keep captures read-only, return finite data,
and do not rely on execution order. Assembly uses a fixed contribution order;
external linear solvers can still introduce platform-dependent rounding.
