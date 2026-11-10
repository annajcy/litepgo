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

`energy.evaluate(x)` creates an independent `EnergyEvaluation` with a copy of
`x`. Construction does not compute value or derivatives. Access prepares and
caches results on demand:

| Python call | Result | Shape |
| --- | --- | --- |
| `e.value` | $E(x)$ | scalar |
| `e.gradient()` | exact gradient | `(n,)` |
| `e.hessian()` | exact Hessian | `(n,n)` |
| `e.projected_hessian()` | symmetric PSD optimization approximation | `(n,n)` |
| `e.max_feasible_step(dx)` | conservative feasible bound | scalar in `[0,1]` |

Here `n = energy.num_dofs`. Vectors are NumPy arrays / C++ `Vec`; matrices are
SciPy CSC / C++ `SpMat`, containing both symmetric triangles. Returned arrays and
matrices own their data. Repeated requests reuse this evaluation's cached results.
Native composition and solver internals read the cached derivatives by const
reference; this does not change the owning results returned by the public C++
and Python methods. Cached storage stays valid for the evaluation's lifetime.

Every evaluation owns an independent input snapshot, including weighted-sum
children. Affine-mapped children own the transformed input `T*x+b`. Changing or
deleting the caller's input, or destroying another evaluation, does not affect
it. The Energy model remains borrowed.

```python
import numpy as np
import scipy.sparse as sp
from litepgo import energy

E = energy.QuadraticEnergy(sp.eye(2, format="csc"))
x = np.array([1.0, 2.0])
e = E.evaluate(x)
H = e.hessian()  # One joint assembly also prepares value and gradient.
g = e.gradient()
assert e.value == 2.5
np.testing.assert_allclose(g, x)
np.testing.assert_allclose(H.toarray(), np.eye(2))
```

```cpp
auto e = E.evaluate(x);
SpMat H = e->hessian();
Vec g = e->gradient();
double value = e->value();
```

Value-only requests do not compute derivatives. Gradient requests also prepare
value; Hessian requests also prepare value and gradient. Asking for value before
Hessian may enter the numerical kernel twice. Exact and projected Hessians have
separate caches; computing a local projection does not necessarily compute the
assembled exact Hessian. Model configuration must stay unchanged while an
evaluation is in use.

## Hessian projection and sparse structure

`hessian()` always returns the mathematical second derivative.
`projected_hessian()` returns a symmetric positive-semidefinite approximation
for optimization. Its default is a symmetric eigendecomposition followed by
clamping negative eigenvalues to zero. It is not an SVD. Newton's diagonal
regularization remains a separate solver operation.

Mesh/FEM and contact energies project local Hessians before assembly. This
preserves their assembly structure but generally differs from projecting the
assembled global Hessian. Custom implementations may provide their own PSD fast
path. Values and gradients always remain exact.

Fixed subclasses snapshot the exact Hessian's symmetric square pattern; dynamic
subclasses generate it at the input when requested. Preserve structural zeros.
The projected Hessian may have a different pattern: default spectral projection
can be dense, with cubic time and quadratic memory costs. Large problems should
provide local or analytic projection paths. A zero Hessian may use an empty
$n\times n$ pattern.

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

The old scalar `eval`, `eval_value`, `eval_gradient`, `eval_hessian` methods and
`HessianKind` enum are removed without compatibility aliases. Create one
evaluation, then request its results. Replace the enum in Newton options with
`project_hessian=True` (default) or `False` for exact curvature. Move the feasible
step query to `energy.evaluate(x).max_feasible_step(dx)`.

Python scalar hooks now receive `eval_values(x, need_gradient, need_hessian)`;
the last argument is a boolean. Exact derivatives remain required. An optional
`eval_projected_values` hook supplies a PSD fast path; otherwise projection is
automatic. See [custom energies](custom.md).

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

with finite symmetric $A\in\mathbb R^{n\times n}$ and finite optional $b\in\mathbb R^n$ (default zero). Its projected Hessian clamps negative eigenvalues and is cached across inputs;
diagonal matrices use an entrywise fast path.

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
PSD leaf projections. If any weight is negative, projected evaluation falls back
to spectral projection of the exact summed Hessian.
`AffineMappedEnergy(E, T, c)` evaluates $F(z)=E(Tz+c)$ and uses

$$
\nabla_zF=T^T\nabla_xE,\qquad \nabla_z^2F=T^THT.
$$

For an inner energy with $n$ DOFs and an outer input of length $k$, $T$ has shape
$(n,k)$ and $c$ has length $n$. The matrix and offset are finite snapshots; an
omitted or empty offset means zero. Rank deficiency is allowed. The feasible-step
bound is the child's bound at $(Tz+c,T\,dz)$.
For `projected_hessian()`, the same pullback is applied to the child's approximation.
This is how contact positions are connected to simulation displacement DOFs.
See the [composition example](../../examples/energy/energy-composition.md).

## Ownership and concurrency

C++ evaluations borrow their Energy, which must outlive them. Python bindings
keep the Energy alive for the evaluation's lifetime. The input is copied, but
model parameters and callback configuration are not snapshotted. Each evaluation
is used by one caller at a time; different evaluations may be interleaved.


`WeightedSumEnergy` and `AffineMappedEnergy` keep Python subclasses alive.
Children must not strongly reference their owner; use `weakref.ref` for a
back-reference. The bindings do not collect cross-language ownership cycles.
See the [Python ownership contract](../concepts.md#python-ownership-must-remain-acyclic),
which also applies to callbacks, contact broad phases, and spatial Response dependencies.

Composites retain their child energies, which may be shared. They consume
child evaluations retained at the corresponding input. Internal assembly plans
are keyed by returned matrix structure. Newton owns symbolic factorization reuse. No allocation-free
evaluation guarantee or public preparation protocol is provided.

C++ Energy evaluation is `const`; implementations may still use mutable workspaces.
The base imposes no lock. `MeshEnergy` guards evaluation and feasible-step
workspaces; `WeightedSumEnergy` and `AffineMappedEnergy` guard their numerical
evaluation caches. Their feasible-step calls use local data and delegate to
children. Overlapping guarded calls raise an error; guards are released on
exceptions. Thread safety also depends on the children.
