# Point embedding

[Back to Spatial queries and embedding](./index.md)

**Problem.** Generate one reference tetrahedron or hex, scaled by `(2,3,4)`,
and sample two known reference points. Locate their physical positions and build
a sparse displacement map. For each point,

$$
X_{e_i}(\xi_i)=p_i,\quad
u_i=N_B(\xi_i)T_{e_i}G_{e_i}u,\quad u_s=Wu,\quad x_s=p+Wu.
$$

Here $u_i$ is the displacement at point $i$.
$G_e$ gathers global physical DOFs and $T_e$ maps them to interpolation
coefficients. Geometry uses Tet P1 or Hex Q1; displacement uses Tet P1, Hex Q1,
or Hex Hermite. Hermite has 24 physical DOFs per vertex on these axis-aligned boxes.

**API walkthrough.** `ElementLocator[GeometryBasis](fields).locate(points)`
returns element IDs and $\xi$. Construct a second `PointEmbedding` from known
IDs/coordinates. `build_point_displacement_map[B]` creates $W$ for each
embedding. Independently gather physical DOFs, apply `BasisDofTransform`, create
an `InterpolatedField3D`, and evaluate it directly. `SurfaceKinematics` then uses
that same $W$ for positions/displacements.

**Expected result.** Three lines, one each for TetP1, HexQ1, and HexHermite,
report `points=2, verified=1`. Assertions compare located and known embeddings
and compare mapped to directly interpolated displacement. The locator rejects
outside points; it does not extrapolate. This example samples independent points;
[Hermite plasticity](../plasticity/hex-hermite-plastic-twist.md) separately demonstrates
uniform boundary sampling.

## Run and inspect

```sh
cmake --build build/release --target point_embedding litepgo_python
build/release/examples/fem/embedding/point_embedding
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/embedding/point_embedding.py
```

::: details Complete Python source — `point_embedding.py`

<<< ../../../examples/fem/embedding/point_embedding.py

:::

::: details Complete C++ source — `point_embedding.cpp`

<<< ../../../examples/fem/embedding/point_embedding.cpp

:::
