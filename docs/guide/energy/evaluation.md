# Energy evaluation and composition

An Energy represents a scalar function $E:D\subseteq\mathbb R^n\to\mathbb R$.
Only the input $x$ is differentiated; model configuration and $n$ remain fixed.
Results must not depend on the order of previous evaluations. Where $E$ is twice
differentiable, its derivatives are

$$
g_i(x)=\frac{\partial E}{\partial x_i},\qquad
H_{ij}(x)=\frac{\partial^2 E}{\partial x_i\partial x_j}.
$$

Dynamic sparsity does not imply differentiability at active-set transitions.

## Evaluate an energy

| Python call | Result | Shape |
| --- | --- | --- |
| `eval_value(x)` | $E(x)$ | scalar |
| `eval_gradient(x)` | $g(x)$ | `(n,)` |
| `eval_hessian(x, kind=HessianKind.Exact)` | $H(x)$ or the requested approximation | `(n,n)` |
| `eval(x, gradient=False, hessian=None)` | value and requested derivatives | triple |
| `max_feasible_step(x, dx)` | conservative feasible bound | scalar in `[0,1]` |

Here `n = num_dofs`, a read-only Python property / C++ query. Inputs and directions
have shape `(n,)`. Vectors are NumPy arrays / C++ `Vec`; matrices are SciPy CSC /
C++ `SpMat`. Hessians contain **both symmetric triangles**, not packed or
triangular storage. After constructing an energy, Python can request:

```python
import numpy as np
import scipy.sparse as sp
from litepgo import energy

E = energy.QuadraticEnergy(sp.eye(2, format="csc"))
x = np.array([1.0, 2.0])
value, gradient, hessian = E.eval(x, gradient=True, hessian=energy.HessianKind.Exact)
assert value == 2.5
np.testing.assert_allclose(gradient, x)
np.testing.assert_allclose(hessian.toarray(), np.eye(2))
```

Unrequested outputs are `None`. The convenience calls are `eval_value`,
`eval_gradient`, and `eval_hessian(x, kind=HessianKind.Exact)`.
`max_feasible_step(x, dx)` bounds the entire segment
$x+t\,dx$, $0\le t\le\alpha$. Start from a feasible state.

C++ returns independently owned results with optional derivatives:

```cpp
auto result = E.eval(x, true, litepgo::energy::HessianKind::Exact);
const double value = result.value;
const auto& g = *result.gradient;
const auto& H = *result.hessian;
```

Unrequested derivatives are `std::nullopt`. Hessian structure is prepared at the
supplied input, only when requested. Joint evaluation invokes one numerical
assembly. Later calls do not modify previous results. The mathematical base has
no concurrency guard; MeshEnergy and composites protect their shared workspaces.

## Hessian kinds and sparse structure

`Exact` returns the mathematical Hessian. `Projected` returns an
implementation-provided approximation for solver steps; it need not be an exact
derivative, a global PSD projection, or even PSD. Values and gradients are
unchanged by this choice. Mesh energies clamp negative eigenvalues of each local
Hessian before assembly; custom energies supply their own approximation.

Fixed subclasses snapshot a full symmetric square pattern. Dynamic subclasses
generate it from the current $x$ when a Hessian is requested. A pattern is a
structural superset for **both** Hessian kinds, including any fill introduced by
projection. Its stored numerical values are ignored. Preserve stored zeros and
do not prune entries merely because their current derivative vanishes.
A zero Hessian may use an empty $n\times n$ pattern.

## Feasibility

For feasible $x$, the returned $\alpha\in[0,1]$ promises

$$
x+t\,dx\in D\quad\text{for every }t\in[0,\alpha].
$$

The endpoint is included. This is a conservative bound, not necessarily the
largest possible step. The caller must start in the domain; the base validates
dimensions and finite inputs but does not establish feasibility of $x$.
The default bound of one is appropriate only for an unrestricted segment.
Mathematical feasibility does not guarantee representable floating-point results;
overflow may still make an evaluation fail. Newton propagates evaluation errors.

## Migration

`WeightedEnergySum` is renamed `WeightedSumEnergy`, including its C++ header
`weighted_sum_energy.hpp`. Remove `update_pattern`, `hessian_pattern`, and
`has_fixed_pattern` calls. Replace native output pointers with the owned result
above. Python retains the tuple-returning `eval`; replace
`eval_projected_hessian(x)` with `eval_hessian(x, kind=HessianKind.Projected)`.
Fixed subclasses now take only `pattern`; dynamic subclasses take only
`num_dofs` and implement `hessian_pattern_at(x)`. No compatibility aliases remain.

## Validation in Debug and Release

Inputs, dimensions, requested outputs, finite values and Python derivative
mapping are always checked. Fixed patterns are validated at construction;
dynamic patterns once per Hessian request. Native numerical hooks must preserve
canonical structure and symmetry: these implementation contracts are checked in
Debug; Release relies on them. Invalid input raises `invalid_argument`, broken
outputs raise `logic_error`, and infeasibility raises `domain_error`. A Python
numerical hook may return `None` for infeasibility; unrelated exceptions propagate.

## Compose built-in energies

`QuadraticEnergy(A, b)` represents

$$
E(x)=\tfrac12x^TAx+b^Tx,\qquad g=Ax+b,\qquad H=A,
$$

with finite symmetric $A\in\mathbb R^{n\times n}$ and finite optional $b\in\mathbb R^n$ (default zero). It returns the same Hessian for both
kinds; an indefinite $A$ is not made PSD by requesting `Projected`.

`VertexPullingEnergy(n, position_dofs, targets, stiffness)` supplies soft springs:

$$
E_{\rm pull}(x)=\frac{k}{2}\sum_i\|x[d_i]-t_i\|^2.
$$

The indices are distinct across all `d_i`, lie in `[0,n)`, and `stiffness` is finite and positive.
Python targets have shape `(number_of_vertices,3)`; C++ takes a flat vector
in vertex/xyz order. Targets use the coordinates of the
input vector: if the input is displacement, pass **anchor minus rest position**.
This is a soft penalty, unlike a hard `DofReduction` constraint.

`WeightedSumEnergy([(E0, w0), ...])` requires at least one term on the same DOFs:

$$
E(x)=\sum_i w_iE_i(x),\qquad g=\sum_i w_i g_i,\qquad H=\sum_i w_i H_i,
\qquad \alpha=\min_i\alpha_i(x,dx).
$$

Its domain is the intersection of the child domains, even for negative weights.
Weights must be finite and nonzero, but may be negative. Positive weights preserve
PSD leaf projections; negative weights can produce an indefinite Hessian.
`AffineMappedEnergy(E, T, c)` evaluates $F(z)=E(Tz+c)$ and uses

$$
\nabla_zF=T^T\nabla_xE,\qquad \nabla_z^2F=T^THT.
$$

For an inner energy with $n$ DOFs and an outer input of length $k$, $T$ has shape
$(n,k)$ and $c$ has length $n$. The matrix and offset are finite snapshots; an
omitted or empty offset means zero. Rank deficiency is allowed. The feasible-step
bound is the child's bound at $(Tz+c,T\,dz)$.
For `Projected`, the same pullback is applied to the child's approximation.
This is how contact positions are connected to simulation displacement DOFs.
See the [composition example](../../examples/energy/energy-composition.md).

## Ownership and concurrency

`WeightedSumEnergy` and `AffineMappedEnergy` keep Python subclasses alive.
Children must not strongly reference their owner; use `weakref.ref` for a
back-reference. The bindings do not collect cross-language ownership cycles.
See the [Python ownership contract](../concepts.md#python-ownership-must-remain-acyclic),
which also applies to callbacks, contact broad phases, and FEM Field dependencies.

Composites retain their child energies, which may be shared. They consume
independent child results and cache only internal assembly plans keyed by returned
matrix structure. Newton owns symbolic factorization reuse. No allocation-free
evaluation guarantee or public preparation protocol is provided.

C++ Energy evaluation is `const`; implementations may still use mutable workspaces.
The base imposes no lock. `MeshEnergy` guards evaluation and feasible-step
workspaces; `WeightedSumEnergy` and `AffineMappedEnergy` guard their numerical
evaluation caches. Their feasible-step calls use local data and delegate to
children. Overlapping guarded calls raise an error; guards are released on
exceptions. Thread safety also depends on the children.
