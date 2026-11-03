# Tet P1 corner pull

[Back to Elastic equilibrium](./index.md)

**Problem.** Read `assets/box-tet.veg` and pull its eight bounding-box corners
toward an enlarged box. For the minimum corner $X_{\min}$,

$$
t_i=0.3(X_i-X_{\min}),\qquad k=1.
$$

The minimum corner's anchor remains at its rest position. Geometry and
displacement are Tet P1, so $F$ is elementwise constant and the centroid rule
integrates this density exactly for each fixed material element.

**API walkthrough.** `read_veg` supplies rest positions and topology.
`VertexDofMap(topology,3)` gathers coordinate and displacement data.
`InterpolatedField3D[TetP1Basis]` constructs rest fields;
`IdentityField3D` supplies $F_p=I$ and $\theta=(2,3)$.
Instantiate `VolumetricFemEnergy[NeoHookean,TetP1Basis,TetCentroidQuadrature]`,
add `VertexPullingEnergy`, and call `newton_solve` on their sum.

**Expected result.** A converged expanded shape, residual within tolerance,
and exported VEG/OBJ. Corners approach but do not exactly meet the anchors:
elastic restoring forces balance the springs. Input must have positive extent
in all axes and exactly the eight expected bounding-box corners.

## Run and inspect

```sh
cmake --build build/release --target tet_p1_corner_pull litepgo_python
build/release/examples/fem/equilibrium/tet_p1_corner_pull
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/equilibrium/tet_p1_corner_pull.py
```

::: details Complete Python source — `tet_p1_corner_pull.py`

<<< ../../../examples/fem/equilibrium/tet_p1_corner_pull.py

:::

::: details Complete C++ source — `tet_p1_corner_pull.cpp`

<<< ../../../examples/fem/equilibrium/tet_p1_corner_pull.cpp

:::
