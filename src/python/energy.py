"""Energies and their composites, mirroring src/core/energy.

ParametricEnergy(state_pattern, parameter_pattern, fixed_pattern=True) defines E(u,p).
energy_at(p) and force_at(u) create Energy/Response views sharing the same gradient
hook; fixed inputs are copied and the model is retained. See guide/energy/parametric.

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

Vector responses and analytic sparse Jacobians.

Subclass FixedPatternResponse(pattern) and override eval_values(x, need_jacobian).
Return (value, Jacobian) when requested, otherwise value; None marks an infeasible
point. Jacobians may be dense or SciPy sparse, but nonzero entries outside the
declared structural pattern are rejected. Public eval methods perform validation.

For changing structure, subclass DynamicPatternResponse(initial_pattern) and also
implement pattern_at(x). Its stored values are ignored; rows/columns must remain
unchanged. Call update_pattern(x) before requesting derivatives. Value-only calls
need no pattern update. Gauss-Newton updates the pattern automatically. Pattern
changes do not imply differentiability at switching points.

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
Their defaults multiply local explicit Jacobians; shared/repeated indices sum.
All Response instances expose eval_jvp(x, dx) and eval_vjp(x, gy). Custom
FixedPatternResponse/DynamicPatternResponse subclasses may override
 eval_jvp_values(x, dx) and eval_vjp_values(x, gy).
Update dynamic patterns before products as before eval_jacobian. Direct products
do not remove the stored Jacobian pattern. No automatic linearization cache is kept.
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

ParametricEnergy = _native.ParametricEnergy
__all__ += ["ParametricEnergy"]
