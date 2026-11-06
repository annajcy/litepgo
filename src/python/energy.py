"""Energies and their composites, mirroring src/core/energy.

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
Check has_hessian first; unsupported calls raise RuntimeError. ExpResponse and compositions with two second-order children support it.

Subclass FixedPatternResponse(pattern, hessian_pattern=None) or
DynamicPatternResponse(initial_pattern, initial_hessian_pattern=None).
Implement eval_impl(x, need_jacobian), returning value or (value, Jacobian).
Return None for infeasibility. Optional eval_hessian_impl(x, weights) returns a
matrix. Omitting the Hessian pattern means unsupported; an empty square pattern
means zero curvature. Dynamic subclasses implement jacobian_pattern_at(x), and
optionally hessian_pattern_at(x), whose default keeps its current structure.
Patterns describe structural supersets, independent of numerical output weights.
Arguments own their data; matrices can be dense or SciPy sparse. Public calls
validate inputs/results and reject concurrent or reentrant calls.

update_patterns(x) explicitly prepares both supported derivative patterns and
reports whether either changed. Pattern accessors return copies; has_fixed_pattern
covers both structures. Native *_into calls use this protocol for buffer reuse;
Python exposes only owning evaluation. See guide/energy/responses for migration.

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

ComposedResponse = _native.ComposedResponse
ElementwiseResponse = _native.ElementwiseResponse
ExpResponse = _native.ExpResponse
__all__ += ["ComposedResponse", "ElementwiseResponse", "ExpResponse"]
