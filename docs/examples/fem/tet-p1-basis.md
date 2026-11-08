# Tet P1 basis and field interpolation

[Back to FEM basics](index.md)

**Problem.** Evaluate the four Tet P1 scalar shape functions at the reference
tetrahedron's centroid $\xi=(1/4,1/4,1/4)$, along with their first and second
reference-coordinate derivatives. Then sample a target function at the four
vertices, build a scalar field, and check that interpolation reconstructs the
target at three interior points.

For $\xi=(r,s,t)$ with $r,s,t\ge0$ and $r+s+t\le1$,

$$
N(\xi)=(1-r-s-t,\ r,\ s,\ t)^T,\qquad
\frac{\partial N}{\partial\xi}=
\begin{bmatrix}-1&-1&-1\\1&0&0\\0&1&0\\0&0&1\end{bmatrix},
\qquad
\frac{\partial^2N}{\partial\xi^2}=0.
$$

**API walkthrough.** Use the basis type directly: `fem.TetP1Basis` in Python
or `litepgo::fem::TetP1Basis` in C++. Read `num_basis_functions`, then call
`eval_N`, `eval_dN_dxi`, and `eval_d2N_dxi2` without constructing an instance.
Rows follow the coefficient order of vertices
$(0,0,0),(1,0,0),(0,1,0),(0,0,1)$.
The [basis guide](../../guide/fem/basis.md#core-api) defines the common contract;
its [Tet P1 formulation](../../guide/fem/basis.md#tet-p1) gives the reference
domain and ordering.

## Sample a target and construct a field

Choose the affine target

$$
u(r,s,t)=2r+3s-t.
$$

Each Tet P1 coefficient is a vertex value. For every basis row $i$, obtain the
reference vertex from `B.coefficient_info(i).point` and evaluate the target:

| Vertex | Target value |
| --- | --- |
| $(0,0,0)$ | 0 |
| $(1,0,0)$ | 2 |
| $(0,1,0)$ | 3 |
| $(0,0,1)$ | -1 |

The scalar field's coefficient matrix is therefore `[[0],[2],[3],[-1]]`,
with shape `(4,1)`. Construct `fem.InterpolatedResponse[B](coefficients)` in
Python or `fem::InterpolatedResponse<B>` in C++. Query `field.eval_value(xi)[0]`
to evaluate the interpolant. Its single component follows

$$
u_h(\xi)=\sum_{i=0}^3N_i(\xi)c_i.
$$

At $\xi=(0.2,0.3,0.1)$, this gives
$0(0.4)+2(0.2)+3(0.3)-1(0.1)=1.2$, equal to the target.
The programs also check $(0.25,0.25,0.25)$ and $(0.1,0.1,0.2)$.
The reconstruction is exact up to floating-point error because the target is
affine, which Tet P1 can represent. A general nonlinear target need not be
reconstructed exactly by this basis.

**Expected result.** Both programs print:

```text
TetP1: functions=4, verified=1
TetP1 field: samples=3, verified=1
```

The programs check that `N` contains four values of $1/4$, `dN` is the
constant `(4,3)` matrix above, and `d2N` is a zero `(4,6)` matrix with columns
`(11,12,13,22,23,33)`. Native checks remain active in Release builds.
The field checks then compare interpolated and target values at all three
interior points with a floating-point tolerance. All coordinates are reference
coordinates; the example applies no physical-coordinate transform or material
integration.

## Run and inspect

Run from the repository root after [configuring the release build](../../guide/build.md):

```sh
cmake --build build/release --target tet_p1_basis litepgo_python
build/release/examples/fem/basis/tet_p1_basis
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/basis/tet_p1_basis.py
```

The C++ target links `litepgo::fem`.

::: details Complete Python source — `tet_p1_basis.py`

<<< ../../../examples/fem/basis/tet_p1_basis.py

:::

::: details Complete C++ source — `tet_p1_basis.cpp`

<<< ../../../examples/fem/basis/tet_p1_basis.cpp

:::
