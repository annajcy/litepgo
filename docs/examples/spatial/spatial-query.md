# Spatial hash queries

[Back to Spatial queries and embedding](./index.md)

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

## Run and inspect

```sh
cmake --build build/release --target spatial_query litepgo_python
build/release/examples/spatial/spatial_query
PYTHONPATH=build/release/python uv run --no-sync python examples/spatial/spatial_query.py
```

::: details Complete Python source — `spatial_query.py`

<<< ../../../examples/spatial/spatial_query.py

:::

::: details Complete C++ source — `spatial_query.cpp`

<<< ../../../examples/spatial/spatial_query.cpp

:::
