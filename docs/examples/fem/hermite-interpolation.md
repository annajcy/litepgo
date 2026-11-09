# Hermite interpolation from coefficient descriptions

[Back to FEM basics](index.md)

**Problem.** Reconstruct $u(\xi)=\sum_{d=1}^D\xi_d^2$ using cubic Hermite
in 1D, Quad Hermite in 2D, and Hex Hermite in 3D. Prepare the coefficients
using the basis descriptions rather than manually numbering derivative slots.
In 2D and 3D, also put these coefficients into an interpolated Response and
compare its values with the target at three interior points.

For `info = B.coefficient_info(i)`, the entry in coefficient row $i$ is

$$
c_i=(\partial^{\alpha_i}u)(p_i),\qquad
p_i=\texttt{info.point},\quad \alpha_i=\texttt{info.derivative\_order}.
$$

For this target polynomial:

- A value sample is $\sum_d(p_i)_d^2$.
- A first-derivative sample in direction $d$ is $2(p_i)_d$.
- All mixed derivative samples are zero.

The supported Hermite coefficient slots use only zero or one derivative per
axis; pure second derivatives are outputs to verify, not coefficient slots.
For example, the 1D coefficient array is $[0,0,1,2]$ for
$[u(0),u'(0),u(1),u'(1)]$.

**API walkthrough.** Query `coefficient_info` for each row, inspect `point`
and the tuple `derivative_order`, and sample the target accordingly. Evaluate
`N`, `dN/dxi`, and `d2N/dxi2` at $\xi_d=0.5$. The corresponding coefficient
contractions must reproduce $u$, $\nabla_\xi u=2\xi$, and the packed Hessian
$2I$. All coordinates and derivatives are reference quantities; physical
Hermite DOFs use a [separate transform](../../guide/fem/hermite.md).

## Reconstruct through a Response

The target is implemented explicitly as `target(point)`. For each corner,
`coefficient_info` specifies whether to fill a value or a derivative slot:
evaluate `target` for value slots, use its analytic first derivative for
first-derivative slots, and fill mixed derivative slots with zero.

For a scalar field, reshape the coefficient vector into one column:
`(16,1)` for Quad Hermite or `(64,1)` for Hex Hermite. Construct
`fem.InterpolatedResponse[B](coefficients)` and call `field.evaluate(query).value[0]`.
The C++ example uses the corresponding `InterpolatedResponse<B>`.
The response owns the sampled coefficients and performs the basis-weighted sum.

The 2D query points are $(0.2,0.35)$, $(0.3,0.45)$, and $(0.4,0.55)$.
The 3D points append coordinates $0.5$, $0.6$, and $0.7$, respectively.
At each query, the examples compare the response value directly with the target
formula. The quadratic target belongs to the bicubic/tricubic interpolation
space, so these values should agree up to floating-point error.

`InterpolatedResponse` supports 1D, 2D and 3D reference coordinates. This example also
shows the equivalent direct basis contraction in 1D.

**Expected result.** Both programs print:

```text
CubicHermite: value=0.25, verified=1
QuadHermite: value=0.50, verified=1
QuadHermite field: samples=3, verified=1
HexHermite: value=0.75, verified=1
HexHermite field: samples=3, verified=1
```

Native checks remain active in Release builds. The example samples this known
polynomial analytically; it does not automatically differentiate arbitrary
Python or C++ functions. See the [basis guide](../../guide/fem/basis.md)
for all formulations, slot orders, and tensor-product construction.

## Run and inspect

```sh
cmake --build build/release --target hermite_interpolation litepgo_python
build/release/examples/fem/basis/hermite_interpolation
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/basis/hermite_interpolation.py
```

::: details Complete Python source — `hermite_interpolation.py`

<<< ../../../examples/fem/basis/hermite_interpolation.py

:::

::: details Complete C++ source — `hermite_interpolation.cpp`

<<< ../../../examples/fem/basis/hermite_interpolation.cpp

:::
