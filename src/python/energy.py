"""Energies and their composites, mirroring src/core/energy.

AffineMappedResponse(A, offset=None) owns a fixed sparse matrix and offset, and evaluates
A @ x + offset. Dense 2-D arrays and SciPy sparse matrices are accepted. Omitted
or empty offset means zero. Jacobian = A, JVP = A @ dx, VJP = A.T @ weights;
weighted Hessians are zero. Rectangular, rank-deficient, and empty maps are valid.

ComposedResponse(outer, inner) evaluates outer(inner(x)) with chained products.
ElementwiseResponse(input_map, transform) concatenates per-element transformed
values and sums shared-DOF VJP contributions. ExpResponse(size) maps log parameters
to positive coefficients. See guide/energy/parameter-transforms.

Hessians are returned as scipy.sparse.csc_matrix. Before assembling a Hessian of
an energy whose pattern is not fixed, call update_pattern(x) at that x. Values are
always finite; evaluating at an infeasible state raises ValueError.

MeshEnergy(map, batch_size=256) subclasses implement eval_batch.
eval_batch(ids, x, need_gradient, need_hessian) returns (values, gradients, Hessians)
with shapes (B,), (B,n), (B*n,n); x is (B,n), including when B=1.
Return None for an infeasible batch (ValueError). Unrequested derivatives may be None. Optional eval_feasible_step_batch(ids, x, dx) returns (B,) bounds.
TBB may invoke hooks concurrently; shared model state must be safe and hook
order is unspecified. Python inputs own their data. Python execution acquires
the GIL, so parallel speedup requires work that releases it.

Numerical entry points are eval_value, eval_gradient, eval_hessian, and
eval_projected_hessian. eval(x, gradient=False, hessian=None) returns the value
and requested derivatives together.

Subclass FixedPatternEnergy(num_dofs, pattern) or
DynamicPatternEnergy(num_dofs, initial_pattern). Override
 eval_values(x, want_gradient, hessian_kind)
returning (value, gradient_or_None, hessian_or_None). hessian_kind is None when
unrequested; otherwise implement Exact or Projected as requested (there is no
automatic projection). Hessians may be dense or SciPy sparse and must fit the
current structural pattern. Return None for an infeasible state; other Python
exceptions propagate. Optional feasible_step(x, dx) returns a bound in [0, 1]
and defaults to 1. Dynamic subclasses must also override pattern_at(x).
Evaluation arguments own their data. Call update_pattern(x) before requesting
a dynamic Hessian; Newton manages this automatically.

Vector responses and analytic sparse derivatives.

Ordinary eval_value(x), eval_jacobian(x), eval_value_and_jacobian(x),
eval_jvp(x, dx), eval_vjp(x, weights), and eval_hessian(x, weights) evaluate at
the supplied input, preparing required patterns automatically. Joint evaluation
returns (value, Jacobian). Direct products need no pattern preparation; their
defaults prepare and multiply the explicit Jacobian.

Response.eval_hessian returns the exact Hessian of weights @ R(x), holding
weights fixed, as a full symmetric (num_inputs, num_inputs) sparse matrix.
Check has_hessian first; unsupported calls raise RuntimeError. AffineMappedResponse,
ExpResponse and compositions with two second-order children support it.

Subclass FixedPatternResponse(pattern, hessian_pattern=None) or
DynamicPatternResponse(num_inputs, num_outputs).
Implement eval_impl(x, need_jacobian), returning value or (value, Jacobian).
Return None for infeasibility. Optional eval_hessian_impl(x, weights) returns a
matrix. A fixed empty square Hessian pattern means zero curvature; omission
means unsupported. Dynamic subclasses implement jacobian_pattern_at(x), and
second-order subclasses override the has_hessian property and hessian_pattern_at(x).
No previous input or current pattern is retained by the dynamic base.
Patterns describe structural supersets, independent of numerical output weights.
Arguments own their data; matrices can be dense or SciPy sparse. Public calls
validate inputs/results. Response has no global concurrency guard; implementations
with shared workspaces, including MeshResponse, reject overlapping evaluation.
Public pattern access/update and native _into methods have been removed.
See guide/energy/responses for the execution contract and migration.

MeshResponse(input_map, output_map, batch_size=256) groups equal-sized
local inputs/outputs. eval_batch(ids, inputs, need_jacobian) receives (B, ni) inputs
and returns (B, no) values, or (values, Jacobians) with Jacobians (B*no, ni).
Batch hooks run concurrently; do not mutate shared state or assume call order.
The assembler uses fixed original element order. batch_size=1 retains batch axes.

MeshEnergy(map, batch_size=256) and MeshResponse(input_map, output_map,
batch_size=256) are subclass bases. Override eval_batch with the corresponding
batch hook signature. MeshEnergy domains override
eval_feasible_step_batch(ids, x, dx). MeshResponse may override
eval_jvp_batch(ids, x, dx) and eval_vjp_batch(ids, x, gy) to return batch products.
Product hooks may return None for an infeasible state, just like value hooks.
Their defaults multiply local explicit Jacobians; shared/repeated indices sum.
All Response instances expose eval_jvp(x, dx) and eval_vjp(x, weights). Custom
FixedPatternResponse/DynamicPatternResponse subclasses may override
 eval_jvp_impl(x, dx) and eval_vjp_impl(x, weights).
Ordinary products prepare patterns only when the implementation needs them. Direct
products do not remove the stored Jacobian pattern. No automatic linearization cache is kept.
"""

from litepgo._core import energy as _native

HessianKind = _native.HessianKind
Energy = _native.Energy
FixedPatternEnergy = _native.FixedPatternEnergy
DynamicPatternEnergy = _native.DynamicPatternEnergy
QuadraticEnergy = _native.QuadraticEnergy
VertexPullingEnergy = _native.VertexPullingEnergy
WeightedEnergySum = _native.WeightedEnergySum
AffineMappedEnergy = _native.AffineMappedEnergy

__all__ = ["FixedPatternEnergy", "DynamicPatternEnergy", "HessianKind", "Energy", "QuadraticEnergy", "VertexPullingEnergy", "WeightedEnergySum", "AffineMappedEnergy"]

from litepgo._core.energy import Response, FixedPatternResponse, DynamicPatternResponse

__all__ += ["Response", "FixedPatternResponse", "DynamicPatternResponse"]

MeshEnergy = _native.MeshEnergy
MeshResponse = _native.MeshResponse
__all__ += ["MeshEnergy", "MeshResponse"]

AffineMappedResponse = _native.AffineMappedResponse
ComposedResponse = _native.ComposedResponse
ElementwiseResponse = _native.ElementwiseResponse
ExpResponse = _native.ExpResponse
__all__ += ["AffineMappedResponse", "ComposedResponse", "ElementwiseResponse", "ExpResponse"]
