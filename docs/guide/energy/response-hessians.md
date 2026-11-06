# Response Hessians

A Response represents $y=R(x)$, with $n$ inputs and $m$ outputs. Its Jacobian
has shape $m\times n$. Each output $R_i$ has an $n\times n$ Hessian; the public
second-derivative interface returns their weighted sum:

$$
H_R(x,w)=\nabla_x^2\bigl(w^TR(x)\bigr)
        =\sum_{i=1}^m w_i\nabla_x^2R_i(x).
$$

The weights are held **fixed** during differentiation. They have length $m$ and
may be positive, negative, or zero. Choosing a unit vector selects one output's
Hessian. This is an exact derivative, not a PSD projection or an average.

## Evaluate

```python
import numpy as np
from litepgo import energy

response = energy.ExpResponse(2)
x = np.array([0.1, -0.2])
weights = np.array([2.0, -3.0])
assert response.has_hessian
H = response.eval_hessian(x, weights=weights)  # scipy.sparse.csc_matrix, (2, 2)
np.testing.assert_allclose(H.toarray(), np.diag(weights * np.exp(x)))
```

C++ uses `has_hessian()` and
`eval_hessian(x, weights)` returning `SpMat`. Both languages return the **full
symmetric matrix**, including both off-diagonal entries. Inputs and weights must
be finite and have the declared dimensions. Unsupported second derivatives raise
`std::logic_error` / Python `RuntimeError`, even when weights are zero.

`AffineMappedResponse`, `ExpResponse`, and compositions whose children both support Hessians implement
this interface. Existing first-order subclasses remain valid and report
`has_hessian == false` unless they explicitly opt in. `ElementwiseResponse`,
`MeshResponse`, FEM force responses, and equilibrium responses currently remain
first-order. This addition does not change the affine Field contract or add an
automatic Energy/Response adapter.

## Composition and the energy chain rule

For $R=f\circ g$ and $z=g(x)$, the product rule gives

$$
H_R(x,w)=J_g(x)^T H_f(z,w)J_g(x)
         +H_g\bigl(x,J_f(z)^Tw\bigr).
$$

`ComposedResponse(outer, inner)` evaluates both terms. The second term is zero
for affine inner maps; the first is zero for affine outer maps. Nonlinear maps
may require both. Child responses can be shared, including a dynamic child used
in both roles: each derivative is evaluated with the pattern for its own input.

For an outer scalar energy $E(x)=W(R(x))$,

$$
\nabla_x E=J_R^T\nabla_yW,
\qquad
\nabla_x^2E=J_R^TH_WJ_R+H_R(x,\nabla_yW).
$$

The first term differentiates the outer gradient; the second differentiates the
inner Jacobian. For example, $R(x)=x^2$ and $W(y)=y^2/2$ give $4x^2+2x^2=6x^2$.
When passing $\nabla_yW$ as weights, hold its current value fixed: its derivative
has already contributed to the first term. These formulas assume twice
continuously differentiable maps near the evaluation point. Dynamic structure
alone does not establish differentiability at switching points.

## Custom responses

For `FixedPatternResponse`, opt in by supplying a Hessian pattern at construction. Omitting it (or passing
Python `None`) means unsupported. A fixed empty $n\times n$ pattern declares a
zero Hessian, as for affine maps, and still requires a numerical hook.

```python
class Square(energy.FixedPatternResponse):
    def __init__(self, size):
        super().__init__(np.eye(size), hessian_pattern=np.eye(size))

    def eval_impl(self, x, need_jacobian):
        value = x * x
        return (value, np.diag(2 * x)) if need_jacobian else value

    def eval_hessian_impl(self, x, weights):
        return np.diag(2 * weights)
```

The Python hook may return a dense matrix or SciPy sparse matrix. Return `None`
for an infeasible point; other exceptions propagate. Arguments are owning copies.
Wrong dimensions, non-finite results, and nonzero entries outside the declared
pattern are rejected in every build. Numerical symmetry is an implementation
contract checked at runtime in Debug; see [validation levels](responses.md#validation-in-debug-and-release). Implementations with shared workspaces protect those workspaces; the mathematical
Response base does not impose a reentry or concurrency guard.

C++ derives from `FixedPatternResponse(jacobian_pattern, hessian_pattern)` and
implements

```cpp
void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights,
                         SpMat& H) override;
```

Overwrite every stored value of `H` without changing its pattern. The validated
public wrapper returns an owning matrix; the hook is not a separate public
evaluation API. No numerical differentiation or automatic differentiation is
used as a fallback.

For dynamic structure, use `DynamicPatternResponse(num_inputs, num_outputs)`.
Override `has_hessian()` in C++ or the `has_hessian` property in Python to declare
second-order support, and implement both `hessian_pattern_at(x)` and
`eval_hessian_impl`. Missing implementations raise an error rather than return zero.

## Sparse pattern contract

The Hessian pattern is a symmetric structural union of all component Hessian
patterns; a conservative superset is allowed. It is independent of `weights`.
An entry does not disappear merely because a weight or derivative happens to
be zero. Fixed subclasses own structure snapshots; dynamic subclasses query the
structure for each evaluation without retaining a current pattern. No public
pattern accessors, update methods, or `_into` protocol are required.

Composition combines the child matrices through the chain rule. Stored structural
entries must be retained even under zero weights or numerical cancellation.
Hessian capability and input/output dimensions remain fixed after construction.
See [Response evaluation](responses.md) for the subclass and execution contracts.

See the paired [Triangle P1 response example](../../examples/fem/response-hessian.md)
for interpolation, a nonlinear parameter field at a fixed query point, and an
explicit energy Hessian calculation. Arbitrary spatial queries and mixed
coordinate/coefficient derivatives are separate from this coefficient-space API.
