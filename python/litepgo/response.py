"""Vector responses and analytic sparse Jacobians.

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
from litepgo._core.response import Response, FixedPatternResponse, DynamicPatternResponse, MeshResponse

__all__ = ["Response", "FixedPatternResponse", "DynamicPatternResponse", "MeshResponse"]
