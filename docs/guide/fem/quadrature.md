# Quadrature

A quadrature rule supplies reference coordinates $\xi_q$ and weights $w_q$:

$$
\int_{\hat\Omega}f(\xi)\,d\xi
\approx\sum_{q=0}^{n_q-1}w_qf(\xi_q).
$$

The rule describes sampling and integration on a reference element. A
[basis](basis.md) supplies shape functions, and a [field](fields.md) supplies
values at those coordinates. Their combination happens in the integrator.

## Core API

All methods are static; no rule instance is needed. C++ checks the interface
through `Quadrature<Q,D>` and the `Quadrature1D/2D/3D` concepts.

| API | Result | Meaning |
| --- | --- | --- |
| `num_points` | Positive compile-time integer | Total number of integration points |
| `points()` | `(num_points,D)` | One reference coordinate per row |
| `weights()` | `(num_points,)` in Python | Weight for each point row |

C++ returns const references to fixed-size double-precision Eigen matrices:
`Matrix<double,num_points,D>` for points and a column vector of length
`num_points` for weights. Python returns independent NumPy array copies.
In 1D, `points()` still has shape `(num_points,1)`; use `points()[:,0]` to
obtain a flat coordinate array. `weights()` is always a flat Python array.

Row `q` and weight `q` belong together. Integration-point indices are
independent of basis-function indices and coefficient slots. The concept
checks interface shapes and return types; the reference domain and accuracy
come from the particular rule.

## Supported rules

Use `GaussLegendre1D<n>` in C++ and `fem.GaussLegendre1D[n]` in Python;
2D/3D names follow the same convention. Both APIs currently support `n=2`
and `n=4`.

| Rule | Reference domain | Total points | Weight sum | Polynomial exactness |
| --- | --- | --- | --- | --- |
| `GaussLegendre1D[n]` | $[0,1]$ | $n$ | 1 | Degree $\le2n-1$ |
| `GaussLegendre2D[n]` | $[0,1]^2$ | $n^2$ | 1 | Degree $\le2n-1$ in each variable |
| `GaussLegendre3D[n]` | $[0,1]^3$ | $n^3$ | 1 | Degree $\le2n-1$ in each variable |
| `TriangleCentroidQuadrature` | $r,s\ge0,\ r+s\le1$ | 1 | $1/2$ | Total degree $\le1$ |
| `TetCentroidQuadrature` | $r,s,t\ge0,\ r+s+t\le1$ | 1 | $1/6$ | Total degree $\le1$ |

The Gauss template parameter is the **number of points per axis**. For
example, `GaussLegendre3D[4]` has 64 points and is exact through degree seven
in each reference coordinate, up to floating-point error.

## Gauss-Legendre in 1D

The two-point rule on $[0,1]$ has

$$
x_0=\frac12-\frac{1}{2\sqrt3},\qquad
x_1=\frac12+\frac{1}{2\sqrt3},\qquad
w_0=w_1=\frac12.
$$

It integrates every polynomial of degree at most three exactly. For example,
$\sum_qw_qx_q^2=1/3$.

For the four-point rule, define

$$
a=\sqrt{\frac{3+2\sqrt{6/5}}7},\qquad
b=\sqrt{\frac{3-2\sqrt{6/5}}7}.
$$

The ordered points and weights are

$$
x=\frac12(1-a,\ 1-b,\ 1+b,\ 1+a)^T,
$$

$$
w=(w_a,w_b,w_b,w_a)^T,\qquad
w_a=\frac{18-\sqrt{30}}{72},\quad
w_b=\frac{18+\sqrt{30}}{72}.
$$

This rule is exact through degree seven. Both built-in rules have increasing
interior points and positive weights whose sum is one.

## Tensor products

`GaussLegendre2D<n>` and `GaussLegendre3D<n>` directly form Cartesian
products of `GaussLegendre1D<n>`, with `n` equal to 2 or 4.
Each class constructs its points and product weights with nested loops.
For 1D points $x_i$ and weights $w_i$:

$$
\begin{aligned}
\xi_{ij}&=(x_i,x_j), & w_{ij}&=w_iw_j,\\
\xi_{ijk}&=(x_i,x_j,x_k), & w_{ijk}&=w_iw_jw_k.
\end{aligned}
$$

The last axis varies fastest: `q=i*n+j` in 2D and
`q=(i*n+j)*n+k` in 3D. Points and weights share this order.
C++ and Python expose these named Gauss rules. Points and weights are
initialized once and cached; C++ returns constant references to the cached
matrices. This is not a guarantee of compile-time evaluation.

For example, a two-point-per-axis 2D rule integrates $r^3s^3$ exactly.
The guarantee concerns the degree in each variable, even though this
monomial's total degree is six.

## Simplex centroid rules

The triangle rule samples its centroid once:

$$
\xi_0=(1/3,1/3),\qquad w_0=1/2.
$$

The tetrahedron rule samples its centroid once:

$$
\xi_0=(1/4,1/4,1/4),\qquad w_0=1/6.
$$

Both rules integrate affine functions exactly. Integrating the constant
function one gives the reference triangle's area or tetrahedron's volume.

## Reference and physical weights

Weights already include reference-domain measure. In particular, triangle
and tetrahedron weights sum to $1/2$ and $1/6$; they are not normalized to
one. Do not multiply by those factors again.

For a positively oriented solid rest-geometry field $X(\xi)$ with
$J_X=\partial X/\partial\xi$, the integrator computes

$$
\int_{\Omega_0}f(X)\,dV
\approx\sum_q w_q\det J_X(\xi_q)\,f(X(\xi_q)).
$$

`VolumetricElement3D[Q](geometry)` performs this geometric weighting and caches
the resulting points and volume weights. Python `points()` returns the reference
coordinates and `weights()` returns the complete physical weights as independent
arrays; C++ `point(q)` exposes `xi` and `w_dV`. For a surface geometry field
$X(r,s)$, the area weight is
$w_q\|X_{,r}\times X_{,s}\|$. See
[integration ownership](fields.md#integration-and-local-material-code) for
how material values and derivatives consume the complete weight.

The rule and the sampled basis/field must refer to the same reference
domain. The exactness guarantee applies to the **complete reference-coordinate
integrand**, including geometric factors. Nonlinear material densities or
nonaffine geometry can prevent exact integration even when the basis is
polynomial. Assess integration accuracy for the quantity being integrated.

## Runnable example

The paired [quadrature integration example](../../examples/fem/quadrature-integration.md)
uses all three dimensions to integrate sums of squared reference coordinates,
then integrates one over a triangle and tetrahedron. It compares every
result with an analytic integral and includes the complete Python/C++ sources.
