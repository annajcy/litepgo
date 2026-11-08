# Fixed coefficients and spatial derivatives

This paired example builds `InterpolatedResponse[TetP1Basis]` from four
fixed vertex positions. At $\xi=(0.2,0.3,0.1)$ it checks

$$
X(\xi)=(0.4,0.9,0.4),\qquad J_X=\operatorname{diag}(2,3,4),\qquad H_X=0.
$$

It then interpolates $f(\xi)=\xi_1\xi_2$ and $g(\xi)=\xi_2\xi_3$ with Hex Q1.
For fixed weights $(2,-3)$, the exact weighted spatial Hessian is

$$
\nabla_\xi^2(2f-3g)=
\begin{pmatrix}0&2&0\\2&0&-3\\0&-3&0\end{pmatrix}.
$$

The mixed entries are mirrored without an extra factor of two. These derivatives
are with respect to the query coordinate, not the fixed coefficients.
See the [API and formulation](../../guide/fem/interpolated-response.md).

## Run

```sh
cmake --build build/release --target interpolated_response litepgo_python
build/release/examples/fem/basis/interpolated_response
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/basis/interpolated_response.py
```

Both programs perform numerical checks and print:

```text
spatial geometry and weighted Hessian: verified
```

The C++ checks remain active in Release builds.

::: details Python source
<<< ../../../examples/fem/basis/interpolated_response.py
:::

::: details C++ source
<<< ../../../examples/fem/basis/interpolated_response.cpp
:::
