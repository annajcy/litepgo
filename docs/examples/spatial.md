# Box queries and displacement embedding

These examples do not solve an optimization problem. They validate spatial-query
and interpolation operators independently, so failures can be separated from
energy or solver issues. Data is generated in memory and no files are written.

All commands run from the repository root after [configuring the release build](../guide/build.md). Python commands explicitly select that build. Expand the source sections for complete programs, including imports, setup, checks, and output handling.

## Spatial hash queries

**Problem.** Index two boxes

$$
B_0=[(0.2,0.2,0),(1.8,0.8,1)],\qquad
B_1=[(1.7,0.2,0),(1.9,0.8,1)].
$$

Boxes overlap when their closed intervals overlap on each coordinate axis.
The region `[(0.9,0.3,0.2),(1.2,0.6,0.8)]` intersects only box 0. The point
`(1.75,0.5,0.5)` is inside both, and the unique overlapping pair is `(0,1)`.

**API walkthrough.** Construct `spatial.AABB` values and
`spatial.SpatialHash(boxes,cell_size=1)`. Call `query` with a region or a zero-size
point box. `overlapping_pairs(filter)` applies the ID predicate and returns unique
pairs with smaller ID first. The input geometry is snapshotted.

**Expected result.** `region_hits=1, point_hits=2, pairs=1`. The examples assert
IDs as well as counts. Query output order is not generally a contract; the Python
checks sort multi-hit IDs. AABBs are broad spatial bounds, not triangle-contact
or element-containment tests.

### Run and inspect

```sh
cmake --build build/release --target spatial_query litepgo_dev
build/release/examples/spatial/spatial_query
PYTHONPATH=build/release/python uv run --no-sync python examples/spatial/spatial_query.py
```

::: details Complete Python source — `spatial_query.py`

<<< ../../examples/spatial/spatial_query.py

:::

::: details Complete C++ source — `spatial_query.cpp`

<<< ../../examples/spatial/spatial_query.cpp

:::


## Point embedding

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
[Hermite plasticity](plasticity.md#hex-hermite-plastic-twist) separately demonstrates
uniform boundary sampling.

### Run and inspect

```sh
cmake --build build/release --target point_embedding litepgo_dev
build/release/examples/fem/embedding/point_embedding
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/embedding/point_embedding.py
```

::: details Complete Python source — `point_embedding.py`

<<< ../../examples/fem/embedding/point_embedding.py

:::

::: details Complete C++ source — `point_embedding.cpp`

<<< ../../examples/fem/embedding/point_embedding.cpp

:::
