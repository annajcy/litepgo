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
values and sums shared-DOF VJP contributions. Each evaluation retains its per-element
child states across value and derivative calls. ExpResponse(size) maps log parameters
to positive coefficients. See guide/energy/parameter-transforms.

Energy.evaluate(x) returns an EnergyEvaluation borrowing the model and owning a
copy of x; Python keeps the model alive. value, gradient(), hessian(), and
projected_hessian() compute lazily and cache results. Hessian requests also
prepare exact value and gradient. Returned vectors and CSC matrices own storage.
Exact and projected Hessians are separate; projected_hessian() promises a PSD
optimization approximation, with default symmetric spectral clamping. Mesh/FEM
and contact use local projection, which can differ from global projection.
Default global projection may produce a dense matrix. max_feasible_step(dx)
bounds the entire closed segment from the evaluation's input conservatively.

Subclass FixedPatternEnergy(pattern) or DynamicPatternEnergy(num_dofs).
Implement eval_values(x, need_gradient, need_hessian), returning
(value, gradient_or_None, hessian_or_None). The exact Hessian fits the declared
pattern. Optional eval_projected_values with the same signature returns a PSD
approximation and may have a different pattern. Omission uses spectral projection.
Return None for infeasibility; unrelated exceptions propagate. Dynamic subclasses
implement hessian_pattern_at(x). Optional feasible_step(x, dx) defaults to 1.
Model and callback configuration must stay fixed while evaluations are in use.
Each evaluation is used by one caller at a time. Implementations with shared
workspaces reject overlapping computation; batch hooks may run concurrently.

MeshEnergy(map, batch_size=256) implements eval_batch(ids, x, need_gradient,
need_hessian), returning shapes (B,), (B,n), (B*n,n). Optional
 eval_feasible_step_batch(ids, x, dx) returns (B,) conservative bounds.

Ownership must remain acyclic: children must not strongly reference their owner,
including through closures or model containers. Use weakref.ref(owner) when a
back-reference is needed; native Energy and Response instances support weak
references. Bindings keep children alive but do not collect cross-language cycles.
The same contract applies to contact and FEM dependencies.

Vector responses and analytic sparse derivatives.

Response.evaluate(x) returns an independent ResponseEvaluation bound to a copy of x.
Its value property, jacobian(), jvp(direction), vjp(weights), and hessian(weights)
prepare results on demand. Values are retained; equilibrium evaluations also retain
the converged state and prepare exact stiffness once for subsequent products.
Python results own their data. Do not mutate model/callback configuration while an
evaluation is in use. Each evaluation is used by one caller at a time.

ResponseEvaluation.hessian returns the exact Hessian of weights @ R(x), holding
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
Response evaluations expose jvp(dx) and vjp(weights). Custom
FixedPatternResponse/DynamicPatternResponse subclasses may override
 jvp_at(x, dx) and vjp_at(x, weights).
Ordinary products prepare patterns only when the implementation needs them. Direct
products do not remove the stored Jacobian pattern. Evaluation state belongs to a single copied input.
"""

from litepgo._core import energy as _native

EnergyEvaluation = _native.EnergyEvaluation
Energy = _native.Energy
FixedPatternEnergy = _native.FixedPatternEnergy
DynamicPatternEnergy = _native.DynamicPatternEnergy
QuadraticEnergy = _native.QuadraticEnergy
VertexPullingEnergy = _native.VertexPullingEnergy
WeightedSumEnergy = _native.WeightedSumEnergy
AffineMappedEnergy = _native.AffineMappedEnergy

__all__ = ["FixedPatternEnergy", "DynamicPatternEnergy", "EnergyEvaluation", "Energy", "QuadraticEnergy", "VertexPullingEnergy", "WeightedSumEnergy", "AffineMappedEnergy"]

from litepgo._core.energy import Response, ResponseEvaluation, FixedPatternResponse, DynamicPatternResponse

__all__ += ["Response", "ResponseEvaluation", "FixedPatternResponse", "DynamicPatternResponse"]

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
