# Parameter transforms with responses

Use ordinary `Response` objects to convert optimization parameters into physical
material coefficients. The field and material interfaces stay unchanged:

```text
p → input DofMap → element parameters q → local Response → physical coefficients c
```

`energy.ElementwiseResponse(input_map, transform)` snapshots the map's indices,
retains a local Response, and concatenates its outputs in element order. For `N`
elements and a transform with `k` outputs, the result is a vector of length `N*k`.
It does not sum physical values at shared input indices. Its VJP does sum all
contributions to shared or repeated input DOFs.

- `ConstantDofMap` shares the optimization parameters between elements.
- `ElementDofMap` gives each element independent optimization parameters.
- `VertexDofMap` and `StencilDofMap` also work when each local input width matches
  the transform. The transform itself determines how those local inputs are used.

`energy.ComposedResponse(outer, inner)` represents `outer(inner(p))`. Both classes
provide explicit Jacobians and direct JVP/VJP composition. For `c = inner(p)`:

$$
Jv = J_{outer}(c)(J_{inner}(p)v),\qquad
J^T w = J_{inner}(p)^T(J_{outer}(c)^T w).
$$

`ComposedResponse` also implements exact weighted Hessians when both children
support them. `AffineResponse` and `ExpResponse` support second derivatives; `ElementwiseResponse`
and the FEM-specific responses currently remain first-order. See
[Response Hessians](response-hessians.md) for the two-term chain rule.

Products call the children’s products, without assembling either global numerical
Jacobian. Children that do not override products still use their normal explicit
Jacobian fallback. Native FEM mesh responses may form local element Jacobians.

## Affine maps

`energy.AffineResponse(A, offset=None)` directly represents

$$
y=Ax+b,\qquad J=A,\qquad Jv=Av,\qquad J^Tw=A^Tw,\qquad H_R(x,w)=0.
$$

For $A\in\mathbb R^{m\times n}$, inputs have length $n$, outputs have length
$m$, and every weighted Hessian is an empty sparse $n\times n$ matrix.
`has_hessian` is true; its declared derivative structures are fixed. Rectangular, rank-deficient,
zero, and zero-dimensional maps are allowed; no inverse is required.

```python
import numpy as np
from litepgo import energy

A = np.array([[1.0, 0.0, -2.0], [0.5, 3.0, 0.0]])
r = energy.AffineResponse(A, offset=np.array([4.0, -1.0]))
x = np.array([1.0, 2.0, -3.0])
np.testing.assert_allclose(r.eval_value(x), [11.0, 5.5])
np.testing.assert_array_equal(r.eval_jacobian(x).toarray(), A)
assert r.eval_hessian(x, weights=np.ones(2)).nnz == 0
```

Python accepts dense two-dimensional arrays or SciPy sparse matrices/arrays,
using the existing sparse conversion (duplicate entries are summed). The object
owns snapshots of $A$ and $b$; changing the constructor arguments later does not
change the response. The matrix is stored sparsely. Stored zeros are retained
in its fixed Jacobian pattern. An omitted, `None`, or empty offset means zero;
a nonempty offset must have length $m$. Matrix entries and offsets must be finite.
Non-finite numerical results, including overflow in JVP/VJP, raise a domain error
(`ValueError` in Python). Input and direction checks follow the Response API.

C++ accepts a `SpMat` and an optional `Vec` (an empty vector means zero):

```cpp
#include "litepgo/energy/response/affine_response.hpp"

Eigen::Matrix<double, 2, 3> A;
A << 1, 0, -2, 0.5, 3, 0;
litepgo::energy::AffineResponse r(A.sparseView(), Eigen::Vector2d(4, -1));
const auto y = r.eval_value(Eigen::Vector3d(1, 2, -3));
```

Unlike `AffineMappedEnergy(E, T, offset)`, this class has no inner function:
it returns the affine map itself. Use `ComposedResponse(outer, affine)` for
`outer(Ax+b)`, or `ComposedResponse(affine, inner)` for `A*inner(x)+b`.
The [Triangle P1 response example](../../examples/fem/response-hessian.md)
uses this built-in map for sampling and composes it with `ExpResponse`.
It composes ordinary Responses without changing the Basis API.

## Plastic parameters

`fem.LogDiagonalFpResponse(3)` maps three log stretches to the nine column-major
entries of `diag(exp(q))`. With `num_stretches=2`, the third diagonal entry is one.
Both reject overflow and underflow to zero. They enforce positive diagonal
stretches, not volume preservation or a plastic evolution law.

```python
from litepgo import dof, energy, fem

parameter_map = dof.ConstantDofMap(topology, 3)
physical_map = dof.ElementDofMap(topology, 9)
plastic = energy.ElementwiseResponse(parameter_map, fem.LogDiagonalFpResponse())

# ElasticEnergy and ForceFp are the usual selected FEM template instantiations.
def make_energy(p):
    packed = plastic.eval_value(p).reshape(-1, 9)
    fields = [energy.AffineResponse(np.zeros((9, 3)), row) for row in packed]
    return ElasticEnergy(displacement_map, geometry, fields, theta)

def make_response(u):
    force = ForceFp(physical_map, displacement_map, geometry, u, theta)
    return energy.ComposedResponse(force, plastic)
```

Pass these two factories to `StaticEquilibriumResponse` as usual. The same
`plastic` object supplies values for the energy and the parameter chain rule for
the force response. For shells, reshape each nine-entry row to a column-major
3×3 matrix and pass the resulting per-face list to `ShellFemEnergy`.

The complete native and Python examples are
[volume plastic fitting](../../examples/fitting/volume-equilibrium-fp-fit.md) and
[shell plastic fitting](../../examples/fitting/shell-equilibrium-fp-fit.md).

## Material parameters

`energy.ExpResponse(size)` maps a vector componentwise to positive coefficients.
Use the identical arrangement for material parameters:

```python
material = energy.ElementwiseResponse(
    dof.ConstantDofMap(topology, 2), energy.ExpResponse(2))
physical_map = dof.ElementDofMap(topology, 2)

# Value path: material.eval_value(p).reshape(-1, 2) → theta field list.
# Derivative path, with the same fixed Fp fields:
force = ForceMaterial(physical_map, displacement_map, geometry, u, fixed_fp)
response = energy.ComposedResponse(force, material)
```

Only choose exponentiation when positivity is appropriate for those physical
coefficients. It is not a universal admissibility rule for all material laws.
You can also compose a global measurement residual with `ExpResponse`, as in the
[volume material Torch example](../../examples/torch/volume-equilibrium-material-fit-torch.md).
Custom transforms use the existing fixed/dynamic Response subclass APIs.

## C++ and ownership

C++ exposes the same constructor order:

```cpp
auto plastic = std::make_shared<energy::ElementwiseResponse>(
    dof::ConstantDofMap(topology, 3),
    std::make_shared<fem::LogDiagonalFpResponse>(3));
auto response = std::make_shared<energy::ComposedResponse>(force, plastic);
```

Headers are `litepgo/energy/response/{elementwise_response,composed_response,exp_response}.hpp`
and `litepgo/fem/parameter/log_diagonal_fp_response.hpp`. Child responses are
shared-owned; bindings keep Python subclasses alive. Child-to-owner back-references
must be weak: cross-language ownership cycles are not collected by the bindings.
See the [ownership contract](../concepts.md#python-ownership-must-remain-acyclic). Captured model configuration must remain fixed. Thread safety depends on each child implementation; children with shared
workspaces protect their own evaluation.

Elementwise evaluation calls the local Response sequentially, including Python
subclass overrides. It is intended for small local parameter transforms; a batched
implementation may be appropriate for very large parameter fields.

## Sparse patterns and limits

Ordinary evaluation automatically prepares the required dynamic patterns.
All callers use ordinary evaluation and own the returned matrices; see
[Response evaluation](responses.md). A composition's pattern is the structural product of its
children's patterns. A dynamic child may be reused in both composition roles;
each role evaluates its derivatives at its own input. Elementwise responses declare full local blocks in a fixed
sparse gather/scatter structure, and refresh dynamic local child patterns at each
local input. Sparse patterns are still explicitly stored even for product-only use.

This parameterization transforms physical **coefficients**. It does not change the
affine coefficient-map contract, implement `exp(interpolated log field)` semantics,
or supply automatic differentiation. The two equilibrium factories still must
represent the same total potential, including any parameter-dependent loads.
