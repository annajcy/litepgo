# Energy evaluation and composition

An Energy is a scalar function $E:\mathbb R^n\to\mathbb R$ with gradient, sparse
Hessian, and a feasible-step bound. Composition combines such functions while
preserving the derivative chain rule.

## Evaluate an energy

An energy supplies its value, gradient, Hessian pattern, and a feasible-step bound.
After constructing it, Python can request outputs independently:

```python
import numpy as np
import scipy.sparse as sp
from litepgo import energy

E = energy.QuadraticEnergy(sp.eye(2, format="csc"))
x = np.array([1.0, 2.0])
E.update_pattern(x)
value, gradient, hessian = E.eval(x, gradient=True, hessian=energy.HessianKind.Exact)
assert value == 2.5
np.testing.assert_allclose(gradient, x)
np.testing.assert_allclose(hessian.toarray(), np.eye(2))
```

Unrequested outputs are `None`. The convenience calls are `eval_value`,
`eval_gradient`, `eval_hessian` (exact), and `eval_projected_hessian`.
`max_feasible_step(x, dx)` bounds the entire segment
$x+t\,dx$, $0\le t\le\alpha$. Start from a feasible state.

C++ accepts independently nullable gradient and Hessian pointers. The caller
copies the current pattern before Hessian evaluation:

```cpp
E.update_pattern(x);
litepgo::Vec g;
litepgo::SpMat H = E.hessian_pattern();
const double value = E.eval(x, &g, &H, litepgo::energy::HessianKind::Exact);
```

Re-copy `hessian_pattern()` whenever `update_pattern` returns true. Native
assembly overwrites stored values and preserves structure. A stale or incompatible
buffer is an error. Values and gradients do not need a pattern refresh.

## Compose built-in energies

`QuadraticEnergy(A, b)` represents

$$
E(x)=\tfrac12x^TAx+b^Tx,\qquad g=Ax+b,\qquad H=A,
$$

with symmetric $A$ and optional $b=0$. It returns the same Hessian for both
kinds; an indefinite $A$ is not made PSD by requesting `Projected`.

`VertexPullingEnergy(n, position_dofs, targets, stiffness)` supplies soft springs:

$$
E_{\rm pull}(x)=\frac{k}{2}\sum_i\|x[d_i]-t_i\|^2.
$$

Each `d_i` lists three distinct scalar DOFs. Targets use the coordinates of the
input vector: if the input is displacement, pass **anchor minus rest position**.
This is a soft penalty, unlike a hard `DofReduction` constraint.

`WeightedEnergySum([(E0, w0), ...])` sums values and derivatives on the same DOFs.
Weights must be finite and nonzero, but may be negative. Positive weights preserve
PSD leaf projections; negative weights can produce an indefinite Hessian.
`AffineMappedEnergy(E, T, c)` evaluates $F(z)=E(Tz+c)$ and uses

$$
\nabla_zF=T^T\nabla_xE,\qquad \nabla_z^2F=T^THT.
$$

This is how contact positions are connected to simulation displacement DOFs.
See the [composition example](../../examples/energy/energy-composition.md).
