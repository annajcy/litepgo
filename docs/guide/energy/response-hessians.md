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
response.update_pattern(x)
H = response.eval_hessian(x, weights=weights)  # scipy.sparse.csc_matrix, (2, 2)
np.testing.assert_allclose(H.toarray(), np.diag(weights * np.exp(x)))
```

C++ uses `has_hessian()`, `hessian_pattern()`, and
`eval_hessian(x, weights)` returning `SpMat`. Both languages return the **full
symmetric matrix**, including both off-diagonal entries. Inputs and weights must
be finite and have the declared dimensions. Unsupported second derivatives raise
`std::logic_error` / Python `RuntimeError`, even when weights are zero.

`ExpResponse` and compositions whose children both support Hessians implement
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

Opt in by supplying a Hessian pattern at construction. Omitting it (or passing
Python `None`) means unsupported. A fixed empty $n\times n$ pattern declares a
zero Hessian, as for affine maps, and still requires a numerical hook.

```python
class Square(energy.FixedPatternResponse):
    def __init__(self, size):
        super().__init__(np.eye(size), hessian_pattern=np.eye(size))

    def eval_values(self, x, need_jacobian):
        value = x * x
        return (value, np.diag(2 * x)) if need_jacobian else value

    def eval_hessian_values(self, x, weights):
        return np.diag(2 * weights)
```

The Python hook may return a dense matrix or SciPy sparse matrix. Return `None`
for an infeasible point; other exceptions propagate. Arguments are owning copies.
Wrong dimensions, non-finite results, asymmetric results, or nonzero entries
outside the declared pattern are rejected. Reentrant/concurrent calls on the
same Response are rejected, as for first-order evaluation.

C++ derives from `FixedPatternResponse(jacobian_pattern, hessian_pattern)` and
implements

```cpp
void eval_hessian_values(const ConstVecRef& x, const ConstVecRef& weights,
                         SpMat& H) const override;
```

Overwrite every stored value of `H` without changing its pattern. The validated
public wrapper returns an owning matrix; the hook is not a separate public
evaluation API. No numerical differentiation or automatic differentiation is
used as a fallback.

For dynamic structure, use
`DynamicPatternResponse(initial_pattern, initial_hessian_pattern)`; Python uses
the same argument names. Override `hessian_pattern_at(x)` when the Hessian union
changes. Its default retains the current Hessian pattern, which also supports a
dynamic Jacobian with a constant Hessian structure.

## Sparse pattern contract

The Hessian pattern is a symmetric structural union of all component Hessian
patterns; a conservative superset is allowed. It is independent of `weights`.
An entry does not disappear merely because a weight or a derivative happens to
be zero at the current point. The pattern accessor contains zero values and the
Python accessor returns a copy.

`update_pattern(x)` prepares both supported derivative patterns and returns true
if **either** changes. `has_fixed_pattern` now describes both supported patterns;
`FixedPatternResponse` promises both are fixed. Existing one-pattern subclasses
keep their previous meaning. Call `update_pattern` on the outermost dynamic
response before derivatives. The composition's Hessian pattern is the structural
union of $P_{J_g}^TP_{H_f}P_{J_g}$ and $P_{H_g}$, with no numerical cancellation.

The public update validates both candidate patterns before installing either.
Custom dynamic hooks must declare all required entries, and numerical hooks must
reject evaluations requiring entries outside the prepared pattern. Hessian
capability and input/output dimensions remain fixed after construction.

See the paired [Triangle P1 response example](../../examples/fem/response-hessian.md)
for interpolation, a nonlinear parameter field at a fixed query point, and an
explicit energy Hessian calculation. Arbitrary spatial queries and mixed
coordinate/coefficient derivatives are separate from this coefficient-space API.
