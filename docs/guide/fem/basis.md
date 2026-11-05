# Basis functions

A basis supplies scalar shape functions on a reference element. For scalar
coefficients $c_a$,

$$
u(\xi)=\sum_{a=0}^{n_b-1}N_a(\xi)c_a.
$$

The basis defines $N_a$ and its reference-coordinate derivatives; an
[interpolated field](fields.md#interpolate-a-field) owns the coefficients.
For an $m$-component field, each $c_a$ becomes an $m$-component row. Tet P1 has
four scalar basis functions, so a three-component Tet P1 field has twelve
scalar coefficients.

## Core API

All methods are static. In C++, `Basis<B,D>` checks `num_basis_functions`,
`coefficient_info`, and the three evaluation signatures; it does not require inheritance or a basis
instance. For $n_b$ basis functions and reference dimension $D$:

| API | Result | Meaning |
| --- | --- | --- |
| `num_basis_functions` | Positive compile-time integer | Number of scalar shape functions |
| `coefficient_info(i)` | `CoefficientInfo<D>` in C++ | Reference sample defining coefficient row `i` |
| `eval_N(xi)` | `(n_b,)` | Shape-function values |
| `eval_dN_dxi(xi)` | `(n_b,D)` | First reference-coordinate derivatives |
| `eval_d2N_dxi2(xi)` | `(n_b,D*(D+1)/2)` | Packed second reference-coordinate derivatives |

C++ evaluation methods return fixed-size double-precision Eigen vectors and matrices;
Python returns NumPy arrays. The 1D bases take a scalar coordinate; 2D and 3D
bases take coordinate vectors. Values and derivatives can be requested
independently. Every output row uses the same basis-function order as the
field's coefficient rows.

Second-derivative columns use the upper triangle in row order:

$$
\begin{array}{c|l}
D & \text{column order}\\ \hline
1 & (11)\\
2 & (11,12,22)\\
3 & (11,12,13,22,23,33)
\end{array}
$$

For example, column `1` in 3D is
$\partial^2N_a/\partial\xi_1\partial\xi_2$. Mixed entries are stored once
without a factor of two. All derivatives are with respect to reference
coordinates; [physical derivative DOFs](hermite.md) require a separate transform.
Built-in polynomial bases extrapolate outside their reference domains without
clamping. Choose [quadrature](fields.md#integration-applies-the-measure-once)
on the same reference domain.

## Coefficient descriptions

`coefficient_info(i)` describes how to obtain coefficient $c_i$ from a target
function. It returns a snapshot with two fields:

| Field | C++ | Python | Meaning |
| --- | --- | --- | --- |
| `point` | `double` in 1D, fixed-size Eigen vector in 2D/3D | Scalar or NumPy vector | Reference location of the sample |
| `derivative_order` | `std::array<int,D>` | Tuple of `D` integers | Number of reference derivatives along each axis |

Writing $p_i$ for `point` and $\alpha_i$ for `derivative_order`,

$$
c_i=(\partial^{\alpha_i}u)(p_i).
$$

All-zero orders mean a function value. In 3D, `(1,0,0)` means $u_{,r}$,
`(0,1,0)` means $u_{,s}$, and `(1,1,0)` means $u_{,rs}$.
For a vector field, apply this rule to every component of coefficient row $i$.
These derivatives have no factorial or physical-length scaling.

For the 1D cubic Hermite basis:

| `i` | `point` | `derivative_order` | Coefficient |
| --- | --- | --- | --- |
| 0 | `0.0` | `(0,)` | $u(0)$ |
| 1 | `0.0` | `(1,)` | $u'(0)$ |
| 2 | `1.0` | `(0,)` | $u(1)$ |
| 3 | `1.0` | `(1,)` | $u'(1)$ |

The location can repeat while the derivative order distinguishes the samples.
Python exposes the read-only result types as `CoefficientInfo1D`,
`CoefficientInfo2D`, and `CoefficientInfo3D`. C++ uses `CoefficientInfo<D>`.
Indices must be in `[0, num_basis_functions)`; invalid indices throw
`std::out_of_range` in C++ and raise `IndexError` in Python.

The [Hermite interpolation example](../../examples/fem/hermite-interpolation.md)
uses these descriptions to prepare coefficients for a known polynomial and
check its interpolated values, gradients, and Hessians in 1D, 2D, and 3D.
Its 2D/3D cases also construct Fields and compare their values directly with
the target at interior points.
This query describes coefficient samples; `eval_N` and the two derivative
methods evaluate an interpolant after its coefficients are known.
Custom C++ interpolation bases must also provide `coefficient_info` to satisfy
the `Basis` concept.

## Supported bases

| Basis | Reference domain | Scalar functions | Coefficients |
| --- | --- | --- | --- |
| `LinearLagrangeBasis` | $[0,1]$ | 2 | Endpoint values |
| `CubicHermiteBasis` | $[0,1]$ | 4 | Endpoint values and first derivatives |
| `TriangleP1Basis` | $r,s\ge0,\ r+s\le1$ | 3 | Vertex values |
| `TetP1Basis` | $r,s,t\ge0,\ r+s+t\le1$ | 4 | Vertex values |
| `HexQ1Basis` | $[0,1]^3$ | 8 | Corner values |
| `QuadHermiteBasis` | $[0,1]^2$ | 16 | Corner values and derivative slots |
| `HexHermiteBasis` | $[0,1]^3$ | 64 | Corner values and derivative slots |

Python exposes all seven named bases in this table. Generic
`TensorProductBasis2D<B>` / `TensorProductBasis3D<B>` remain C++ building blocks.

## Linear Lagrange in 1D

On $t\in[0,1]$, `LinearLagrangeBasis` uses endpoint order $0,1$:

$$
\ell_0(t)=1-t,\qquad \ell_1(t)=t,\qquad
u(t)=u(0)\ell_0(t)+u(1)\ell_1(t).
$$

The derivative vectors are
$\ell'(t)=(-1,1)^T$ and $\ell''(t)=(0,0)^T$.
These functions generate Hex Q1 by tensor products.

## Triangle P1

On $r,s\ge0,\ r+s\le1$, vertices are ordered
$(0,0),(1,0),(0,1)$. `TriangleP1Basis` returns

$$
N(r,s)=
\begin{bmatrix}1-r-s\\r\\s\end{bmatrix},
\qquad
\frac{\partial N}{\partial(r,s)}=
\begin{bmatrix}-1&-1\\1&0\\0&1\end{bmatrix}.
$$

Coefficients are the three vertex values. The basis spans affine functions,
and every second derivative is zero.

## Tet P1

On $r,s,t\ge0,\ r+s+t\le1$, vertices are ordered
$(0,0,0),(1,0,0),(0,1,0),(0,0,1)$. `TetP1Basis` returns

$$
N(r,s,t)=
\begin{bmatrix}1-r-s-t\\r\\s\\t\end{bmatrix},
\qquad
\frac{\partial N}{\partial(r,s,t)}=
\begin{bmatrix}
-1&-1&-1\\
1&0&0\\
0&1&0\\
0&0&1
\end{bmatrix}.
$$

Coefficients are the four vertex values. The basis spans affine functions,
and every second derivative is zero.

## Hex Q1

`HexQ1Basis` uses the following corner order on $[0,1]^3$.
`HexHermiteBasis` uses the same order.

| Vertex $v$ | Corner $p_v$ |
| --- | --- |
| 0 | $(0,0,0)$ |
| 1 | $(1,0,0)$ |
| 2 | $(1,1,0)$ |
| 3 | $(0,1,0)$ |
| 4 | $(0,0,1)$ |
| 5 | $(1,0,1)$ |
| 6 | $(1,1,1)$ |
| 7 | $(0,1,1)$ |

For $p_v=(a,b,c)$, each shape function is

$$
N_v(r,s,t)=\ell_a(r)\ell_b(s)\ell_c(t).
$$

In the listed order, the complete vector is

$$
N(r,s,t)=
\begin{bmatrix}
(1-r)(1-s)(1-t)\\
r(1-s)(1-t)\\
rs(1-t)\\
(1-r)s(1-t)\\
(1-r)(1-s)t\\
r(1-s)t\\
rst\\
(1-r)st
\end{bmatrix}.
$$

Coefficients are corner values. This spans polynomials of degree at most one
in each variable, including $rs$, $rt$, $st$, and $rst$. Pure second
derivatives vanish, but mixed second derivatives can be nonzero.

## Cubic Hermite in 1D

On $t\in[0,1]$, define $h_{p,d}(t)$ for endpoint $p\in\{0,1\}$ and coefficient
derivative order $d\in\{0,1\}$:

$$
\begin{aligned}
h_{0,0}(t)&=2t^3-3t^2+1,&
h_{0,1}(t)&=t^3-2t^2+t,\\
h_{1,0}(t)&=-2t^3+3t^2,&
h_{1,1}(t)&=t^3-t^2.
\end{aligned}
$$

`CubicHermiteBasis` returns them in order
$[h_{0,0},h_{0,1},h_{1,0},h_{1,1}]$, so

$$
u(t)=h_{0,0}(t)u(0)+h_{0,1}(t)u'(0)
     +h_{1,0}(t)u(1)+h_{1,1}(t)u'(1).
$$

| Function | First derivative | Second derivative |
| --- | --- | --- |
| $h_{0,0}$ | $6t^2-6t$ | $12t-6$ |
| $h_{0,1}$ | $3t^2-4t+1$ | $6t-4$ |
| $h_{1,0}$ | $-6t^2+6t$ | $-12t+6$ |
| $h_{1,1}$ | $3t^2-2t$ | $6t-2$ |

The endpoint conditions are

$$
h_{p,d}^{(k)}(q)=\delta_{pq}\delta_{dk},
\qquad p,q,d,k\in\{0,1\},
$$

where the superscript is the derivative order and $\delta$ is the Kronecker
delta. Value slots have unit value and zero first derivative at their endpoint;
derivative slots have zero value and unit first derivative.

## Quad Hermite

`QuadHermiteBasis` spans bicubic polynomials on $[0,1]^2$.
Corners are ordered $(0,0),(1,0),(1,1),(0,1)$. At each corner $p_v$, the four
coefficient slots are

| Slot $j$ | Derivative index $\alpha_j$ | Scalar coefficient |
| --- | --- | --- |
| 0 | $(0,0)$ | $u(p_v)$ |
| 1 | $(1,0)$ | $u_{,r}(p_v)$ |
| 2 | $(0,1)$ | $u_{,s}(p_v)$ |
| 3 | $(1,1)$ | $u_{,rs}(p_v)$ |

For corner $p_v=(a,b)$ and derivative index
$\alpha_j=(d_r,d_s)$ of slot $j$, the shape function in row $4v+j$ is

$$
N_{4v+j}(r,s)=h_{a,d_r}(r)h_{b,d_s}(s).
$$

This specifies all sixteen functions and their order. At corner $(0,0)$,
the first four functions are $h_{0,0}(r)h_{0,0}(s)$,
$h_{0,1}(r)h_{0,0}(s)$, $h_{0,0}(r)h_{0,1}(s)$, and
$h_{0,1}(r)h_{0,1}(s)$.
A vector field with $m$ components stores coefficients as `(16,m)`.

## Hex Hermite

`HexHermiteBasis` spans tricubic polynomials on $[0,1]^3$, with the eight
corners ordered as for Hex Q1. At each corner $p_v$, the slots are

| Slot $j$ | Derivative index $\alpha_j$ | Scalar coefficient |
| --- | --- | --- |
| 0 | $(0,0,0)$ | $u(p_v)$ |
| 1 | $(1,0,0)$ | $u_{,r}(p_v)$ |
| 2 | $(0,1,0)$ | $u_{,s}(p_v)$ |
| 3 | $(0,0,1)$ | $u_{,t}(p_v)$ |
| 4 | $(1,1,0)$ | $u_{,rs}(p_v)$ |
| 5 | $(1,0,1)$ | $u_{,rt}(p_v)$ |
| 6 | $(0,1,1)$ | $u_{,st}(p_v)$ |
| 7 | $(1,1,1)$ | $u_{,rst}(p_v)$ |

For corner $p_v=(a,b,c)$ and derivative index
$\alpha_j=(d_r,d_s,d_t)$ of slot $j$, the function in row $8v+j$ is

$$
N_{8v+j}(r,s,t)=h_{a,d_r}(r)h_{b,d_s}(s)h_{c,d_t}(t).
$$

This specifies all sixty-four functions and their order. For example,
$N_0=h_{0,0}(r)h_{0,0}(s)h_{0,0}(t)$ and
$N_4=h_{0,1}(r)h_{0,1}(s)h_{0,0}(t)$.
A three-component field has `(64,3)` coefficients.

All Hermite coefficient derivatives are taken with respect to the reference
coordinates, including the third mixed derivative in slot 7. There are no
factorial or physical-length factors in these basis definitions. The
[physical DOF transform](hermite.md#slots-and-physical-to-reference-conversion)
provides the length scales required by the high-level Hex Hermite wrappers.
The third mixed coefficient slot does not add a third-derivative evaluation
method to the core API.

## Tensor-product derivatives and ordering

For a product of 1D factors
$N(\xi)=\prod_{j=1}^D f_j(\xi_j)$, any derivative is

$$
\partial^\beta N(\xi)=\prod_{j=1}^D f_j^{(\beta_j)}(\xi_j).
$$

Choosing $\beta=e_i$, $2e_i$, or $e_i+e_j$ gives the first, pure second,
or mixed second derivative returned by the API. Together with the 1D
polynomials above, this defines all derivative entries for Hex Q1 and both
Hermite bases.

Generic tensor products use one fixed order: reference nodes first, derivative
slots within each node. For endpoint bases, the node order is the Quad/Hex
corner order listed above. Slots use increasing total derivative order,
with earlier coordinate directions first within each degree. Thus 3D Hermite
slots are `u, u_r, u_s, u_t, u_rs, u_rt, u_st, u_rst`.
Values, derivatives, and `coefficient_info` share the same row order.

The index combinations come from the 1D coefficient descriptions. For example,
combining a sample at `point=0, derivative_order=(1,)` with a sample at
`point=1, derivative_order=(0,)` gives `point=(0,1)` and
`derivative_order=(1,0)` in 2D. Evaluation generates rows directly in this
order; there is no separate matrix permutation.

The named tensor bases are C++ aliases:

```cpp
using HexQ1Basis = TensorProductBasis3D<LinearLagrangeBasis>;
using QuadHermiteBasis = TensorProductBasis2D<CubicHermiteBasis>;
using HexHermiteBasis = TensorProductBasis3D<CubicHermiteBasis>;
```

This replaces the generic tensor products' former last-coordinate-fastest row
order. Existing named Quad/Hex coefficient orders are preserved. Code that used
generic tensor products with manually ordered coefficient arrays must update
those arrays; `coefficient_info` supplies the corresponding sample definitions.
Quadrature point ordering is independent and remains last-coordinate-fastest.

## Evaluate a basis

The runnable [Tet P1 basis example](../../examples/fem/tet-p1-basis.md)
evaluates the shape functions and both derivative orders at the reference
centroid. It then samples $u(r,s,t)=2r+3s-t$ at the four vertices, builds an
`InterpolatedField3D`, and checks that the Field reconstructs the target at
interior points. Its page includes run commands, expected results, and complete
Python and C++ sources imported from `examples/fem/basis/`.

Both programs use the basis type directly without constructing an instance.
Python can select the compiled bases above; arbitrary Python-defined bases
cannot be substituted as C++ template arguments.
