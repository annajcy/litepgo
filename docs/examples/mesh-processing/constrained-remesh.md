# Constrained isotropic remeshing

[Back to Mesh processing](./index.md)

Remesh a square triangulated around an off-center vertex. The target edge length
is `0.8`, so the unit boundary edges meet the protected-edge limit of
`4/3 * 0.8`. Pass input vertex ID `4` to hold the interior point at
`(0.2, 0.2, 0)`. The remesher can change interior connectivity.

Both programs find the fixed point by coordinate and compare the boundary
segments by their endpoint coordinates, since output indices may change.
They also run `has_self_intersections` on the result and print the before and
after triangle counts.

## Run and inspect

```sh
cmake --preset release -DLITEPGO_WITH_CGAL=ON
cmake --build build/release --target constrained_remesh litepgo_python
build/release/examples/mesh/constrained_remesh
PYTHONPATH=build/release/python uv run --no-sync python examples/mesh/constrained_remesh.py
```

::: details Complete Python source — `constrained_remesh.py`

<<< ../../../examples/mesh/constrained_remesh.py

:::

::: details Complete C++ source — `constrained_remesh.cpp`

<<< ../../../examples/mesh/constrained_remesh.cpp

:::
