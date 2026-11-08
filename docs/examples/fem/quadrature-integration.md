# Quadrature integration

[Back to FEM basics](index.md)

**Problem.** Compute reference-element integrals directly from integration
points and weights. Use two-point-per-axis Gauss rules in 1D, 2D, and 3D,
then check the reference measures built into the simplex centroid rules.

For $D=1,2,3$, choose the polynomial

$$
f(\xi)=\sum_{d=1}^D\xi_d^2,\qquad
\int_{[0,1]^D}f(\xi)\,d\xi=\frac D3.
$$

Each summand integrates to $1/3$ along its own axis; the remaining axes
each contribute a factor of one. The two-point Gauss rule integrates these
quadratic terms exactly up to floating-point error.

**API walkthrough.** Select a rule type, read `num_points`, then call
`points()` and `weights()`. Python returns arrays with shapes
`(num_points,D)` and `(num_points,)`, including `(num_points,1)` for 1D
points. Evaluate the target once per point row, then form the weighted sum
$\sum_qw_qf(\xi_q)$. C++ uses the same static interface and a loop over `q`.
The [quadrature guide](../../guide/fem/quadrature.md) describes every supported
rule, its formulation, point ordering, and polynomial exactness.

## Reference measures

For the triangle and tetrahedron, use the constant integrand $f=1$:

$$
\int_{\hat T_2}1\,dA=\frac12,\qquad
\int_{\hat T_3}1\,dV=\frac16.
$$

The centroid rules contain those measures in their weights. These examples
apply the same weighted sum without an additional area or volume factor.
All five cases integrate on reference elements. For physical elements,
the [integrator applies the geometric Jacobian](../../guide/fem/quadrature.md#reference-and-physical-weights).

**Expected result.** Both programs print:

```text
GaussLegendre1D[2]: points=2, integral=0.333333, verified=1
GaussLegendre2D[2]: points=4, integral=0.666667, verified=1
GaussLegendre3D[2]: points=8, integral=1.000000, verified=1
TriangleCentroid: points=1, integral=0.500000, verified=1
TetCentroid: points=1, integral=0.166667, verified=1
```

The programs compare results with the analytic values before printing;
native checks remain active in Release builds. Gauss template arguments
specify points per axis: the three cases above use 2, 4, and 8 points in
total. The available Gauss instantiations use 2 or 4 points per axis.

## Run and inspect

Run from the repository root after [configuring the release build](../../guide/build.md):

```sh
cmake --build build/release --target quadrature_integration litepgo_python
build/release/examples/fem/quadrature/quadrature_integration
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/quadrature/quadrature_integration.py
```

The C++ target links `litepgo::fem`.

::: details Complete Python source — `quadrature_integration.py`

<<< ../../../examples/fem/quadrature/quadrature_integration.py

:::

::: details Complete C++ source — `quadrature_integration.cpp`

<<< ../../../examples/fem/quadrature/quadrature_integration.cpp

:::
