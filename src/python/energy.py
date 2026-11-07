"""Energies and their composites, mirroring src/core/energy.

AffineResponse(A, offset=None) owns a fixed sparse matrix and offset, and evaluates
A @ x + offset. Dense 2-D arrays and SciPy sparse matrices are accepted. Omitted
or empty offset means zero. Jacobian = A, JVP = A @ dx, VJP = A.T @ weights;
weighted Hessians are zero. Rectangular, rank-deficient, and empty maps are valid.

IdentityResponse(size) returns its input. ConstantResponse(num_inputs, value)
owns a copy of its output vector. Both accept zero dimensions, support exact
zero Hessians, and evaluate JVPs and VJPs directly.

ComposedResponse(outer, inner) evaluates outer(inner(x)) with chained products.
ElementwiseResponse(input_map, transform) concatenates per-element transformed
values and sums shared-DOF VJP contributions. ExpResponse(size) maps log parameters
to positive coefficients. See guide/energy/parameter-transforms.

Energy evaluates E(x), its gradient (n,), and full symmetric Hessian (n,n).
Hessians are independent scipy.sparse.csc_matrix results. Required structure is
prepared automatically for each input; no public pattern management is needed.
eval_value, eval_gradient, and eval_hessian(x, kind=HessianKind.Exact) return
independent results. eval(x, gradient=False, hessian=None) returns
(value, gradient_or_None, hessian_or_None) in one numerical assembly.
Projected is an implementation-provided solver approximation, not necessarily PSD.
max_feasible_step bounds the entire closed segment conservatively, not maximally.

Subclass FixedPatternEnergy(pattern) or DynamicPatternEnergy(num_dofs).
Implement eval_values(x, want_gradient, hessian_kind), returning the triple above;
return None for infeasibility (public calls raise ValueError). Other exceptions
propagate. Dynamic subclasses implement hessian_pattern_at(x). Optional
feasible_step(x, dx) defaults to 1, appropriate only for unrestricted domains.
Python inputs own their data. Hessians may be dense or sparse and must fit the
declared structure. Fixed structure is snapshotted; dynamic structure is per call.
The mathematical base permits reentry. MeshEnergy and composites with shared
workspaces reject overlapping evaluation. Batch hooks may run concurrently.

MeshEnergy(map, batch_size=256) implements eval_batch(ids, x, need_gradient,
need_hessian), returning shapes (B,), (B,n), (B*n,n). Optional
 eval_feasible_step_batch(ids, x, dx) returns (B,) conservative bounds.

Ownership must remain acyclic: children must not strongly reference their owner,
including through closures or model containers. Use weakref.ref(owner) when a
back-reference is needed; native Energy and Response instances support weak
references. Bindings keep children alive but do not collect cross-language cycles.
The same contract applies to equilibrium callbacks and contact/FEM dependencies.

Vector responses and analytic sparse derivatives.

Ordinary eval_value(x), eval_jacobian(x), eval_value_and_jacobian(x),
eval_jvp(x, dx), eval_vjp(x, weights), and eval_hessian(x, weights) evaluate at
the supplied input, preparing required patterns automatically. Joint evaluation
returns (value, Jacobian). Direct products need no pattern preparation; their
defaults prepare and multiply the explicit Jacobian.

Response.eval_hessian returns the exact Hessian of weights @ R(x), holding
weights fixed, as a full symmetric (num_inputs, num_inputs) sparse matrix.
Check has_hessian first; unsupported calls raise RuntimeError. AffineResponse,
IdentityResponse, ConstantResponse, ExpResponse, and compositions with two
second-order children support it.

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
WeightedSumEnergy = _native.WeightedSumEnergy
AffineMappedEnergy = _native.AffineMappedEnergy

__all__ = ["FixedPatternEnergy", "DynamicPatternEnergy", "HessianKind", "Energy", "QuadraticEnergy", "VertexPullingEnergy", "WeightedSumEnergy", "AffineMappedEnergy"]

from litepgo._core.energy import Response, FixedPatternResponse, DynamicPatternResponse

__all__ += ["Response", "FixedPatternResponse", "DynamicPatternResponse"]

MeshEnergy = _native.MeshEnergy
MeshResponse = _native.MeshResponse
__all__ += ["MeshEnergy", "MeshResponse"]

AffineResponse = _native.AffineResponse
IdentityResponse = _native.IdentityResponse
ConstantResponse = _native.ConstantResponse
ComposedResponse = _native.ComposedResponse
ElementwiseResponse = _native.ElementwiseResponse
ExpResponse = _native.ExpResponse
__all__ += ["AffineResponse", "IdentityResponse", "ConstantResponse", "ComposedResponse", "ElementwiseResponse", "ExpResponse"]
