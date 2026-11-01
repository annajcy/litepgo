"""Energies and their composites, mirroring cpp/litepgo/energy.

Hessians are returned as scipy.sparse.csc_matrix. Before assembling a Hessian of
an energy whose pattern is not fixed, call update_pattern(x) at that x. Values are
always finite; evaluating at an infeasible state raises ValueError.

MeshEnergy(map, eval_batch, batch_size=256, step=None) uses only batch callbacks.
eval_batch(ids, x, need_gradient, need_hessian) returns (values, gradients, Hessians)
with shapes (B,), (B,n), (B*n,n); x is (B,n), including when B=1.
Return None for an infeasible batch (ValueError). Unrequested derivatives may be None. Optional step(ids, x, dx) returns (B,) bounds.
TBB may invoke callbacks concurrently; shared captures must be safe and callback
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

StaticEquilibriumResponse(num_parameters, fixed, initial, options, make_energy, partial)
returns equilibrium displacement and its direct parameter Jacobian. make_energy(p)
returns the total Energy; partial(p, u) returns its sparse gradient/parameter
Jacobian at fixed u. Fixed DOFs and values must be parameter-independent.
Each call starts from initial; sensitivities use the exact unshifted Hessian.
Nonconverged equilibria raise ValueError. See example/fem/fitting/volume_equilibrium_material_fit.py.

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

MeshResponse(input_map, output_map, eval_batch, batch_size=256) groups equal-sized
local inputs/outputs. eval_batch(ids, inputs, need_jacobian) receives (B, ni) inputs
and returns (B, no) values, or (values, Jacobians) with Jacobians (B*no, ni).
Batch callbacks run concurrently; do not mutate shared state or assume call order.
The assembler uses fixed original element order. batch_size=1 retains batch axes.

"""

from litepgo._core import energy as _native

HessianKind = _native.HessianKind
Energy = _native.Energy
FixedPatternEnergy = _native.FixedPatternEnergy
DynamicPatternEnergy = _native.DynamicPatternEnergy
MeshEnergy = _native.MeshEnergy
QuadraticEnergy = _native.QuadraticEnergy
VertexPullingEnergy = _native.VertexPullingEnergy
WeightedEnergySum = _native.WeightedEnergySum
AffineMappedEnergy = _native.AffineMappedEnergy

__all__ = ["FixedPatternEnergy", "DynamicPatternEnergy", "HessianKind", "Energy", "MeshEnergy", "QuadraticEnergy", "VertexPullingEnergy", "WeightedEnergySum", "AffineMappedEnergy"]

from litepgo._core.energy import Response, FixedPatternResponse, DynamicPatternResponse, MeshResponse

__all__ += ["Response", "FixedPatternResponse", "DynamicPatternResponse", "MeshResponse"]

from litepgo._core.energy import StaticEquilibriumResponse

__all__ += ["StaticEquilibriumResponse"]
