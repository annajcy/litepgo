# Spatial interpolation with a Response

`fem.InterpolatedResponse[Basis](coefficients)` evaluates a fixed coefficient
distribution at a reference coordinate. The coefficients are configuration;
**the input and differentiation variable is the reference coordinate**.

For a basis with $k$ functions on $d$ reference coordinates and a finite
coefficient matrix $C\in\mathbb R^{k\times m}$,

$$
R_C(\xi)=C^TN(\xi),\qquad
[J_R(\xi)]_{ij}=\sum_{a=1}^k C_{ai}\frac{\partial N_a}{\partial\xi_j}.
$$

The weighted Hessian follows the ordinary Response contract:

$$
H_R(\xi,w)=\sum_{a=1}^k(C_{a,:}w)\nabla_\xi^2N_a(\xi).
$$

Weights are fixed during differentiation and may be negative or zero.
This is an exact spatial derivative, with no PSD projection.

## API and shapes

Python constructs `fem.InterpolatedResponse[Basis](coefficients=C)`.
C++ constructs `fem::InterpolatedResponse<Basis> r(C)` from
`litepgo/fem/response/interpolated_response.hpp`; link `litepgo::fem`.

| API | Shape / meaning |
| --- | --- |
| Constructor coefficients | `(k,m)`, with `m > 0` |
| `num_inputs` | $d$ |
| `num_outputs` | $m$ |
| `has_hessian` | true, including zero curvature |
| `eval_value(xi)` | `(m,)` |
| `eval_jacobian(xi)` | `(m,d)` |
| `eval_value_and_jacobian(xi)` | value and Jacobian |
| `eval_jvp(xi, direction)` | `(m,)`, direction has length $d$ |
| `eval_vjp(xi, weights)` | `(d,)`, weights have length $m$ |
| `eval_hessian(xi, weights)` | full symmetric `(d,d)` matrix |

Python coordinates are always vectors of shape `(d,)`, **including `(1,)`
for a 1D Response**, whereas the 1D Basis methods take a scalar.
Python Jacobians and Hessians are SciPy CSC matrices; C++ uses `SpMat`.
The implementation uses fixed full structural supersets, retaining stored zeros
even when the actual derivative is zero. JVP/VJP compute products directly.

The object owns a coefficient snapshot. Mutating the source matrix later does
not change it; there is no coefficient setter. Rows follow
`Basis.coefficient_info(i)`: nodal bases store values, while Hermite bases
also store reference-coordinate derivative samples. Columns are output components.
Queries outside the reference domain extrapolate the basis polynomials without
clamping. Inputs and outputs follow the ordinary
[Response validation rules](../energy/responses.md#validation-in-debug-and-release).
All evaluations use call-local data, with no shared mutable workspace.

Python binds all seven existing bases: LinearLagrange, CubicHermite, TriangleP1,
QuadHermite, TetP1, HexQ1 and HexHermite. C++ deduces the reference dimension from
the basis coefficient description and requires the corresponding Basis concept.

## Geometry and coefficient differentiation

For Tet P1, a `(4,3)` matrix of vertex positions is the coefficient matrix.
The response maps reference coordinates to physical positions; its Jacobian
is the geometric coordinate Jacobian. The affine Tet map has zero spatial
Hessians. Hex Q1 is trilinear and can have nonzero mixed spatial derivatives.

This component is useful for fixed geometry queries and spatial material sampling.
It does not impose geometric validity: a geometry consumer must still check
orientation, invertibility and its own element-domain restrictions.

For **coefficient** derivatives at a fixed coordinate, the separate map is

$$
c\mapsto \bigl(N(\xi)^T\otimes I_m\bigr)c,
$$

where $c$ flattens $C$ in row-major order. Use [value_sampling_response](sampling-responses.md), which returns an
`AffineResponse`; it is not the Jacobian of `InterpolatedResponse`.
FEM consumers now use spatial Responses and cache fixed configuration at construction; see [migration and contracts](fields.md).

See the [paired spatial interpolation example](../../examples/fem/interpolated-response.md)
for Tet geometry and a two-output Hex map with a nonzero weighted Hessian.
