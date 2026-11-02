# Spatial queries and embedding

`spatial::SpatialHash` (`litepgo/spatial/spatial_hash.hpp`) owns a snapshot of input
AABBs, indexed in input order. Construct it with boxes and an optional positive cell
size. `query(region)` returns unique IDs of boxes intersecting the inclusive region;
use `{point, point}` for a point query. `overlapping_pairs(filter)` returns unique
accepted pairs with `i < j`; the predicate receives original box IDs and must support
concurrent calls. Const queries can run concurrently. Invalid bounds are rejected. Grid
coordinate overflow throws `std::overflow_error`; excessive box-cell records (including query regions)
throw `std::length_error`. No scanning fallback is performed. The contact spatial hash
adapts primitive-kind filtering to this index. No element locator is provided.

Rest-geometry point location is available as `fem::ElementLocator<GeometryBasis>`
(`litepgo/fem/element_locator.hpp`), with TetP1 and HexQ1 geometry supported.
The locator owns copies of the three-component interpolated X fields, builds a
SpatialHash over their corner bounds, and returns `PointEmbedding.element_ids`
and `PointEmbedding.xi`. The geometry basis is independent of the displacement
basis: Q1 geometry can also be used with Hermite displacement DOFs.

```python
locator = fem.ElementLocator[fem.TetP1Basis](X_fields)
surface = mesh.extract_surface(volume)
embedding = locator.locate(surface.positions)
# X_fields[embedding.element_ids[i]].eval_value(embedding.xi[i]) ~= surface.positions[i]
```

Queries accept finite `(n, 3)` points. Damped Newton inversion checks both physical
residual and reference-domain containment. Bounds use a scale-dependent numerical
padding; reference-domain tolerance is 1e-10. Shared-boundary points select the
lowest successful element ID. Outside points, singular inverse maps, and failed
inverse solves raise an error identifying the point; no projection or extrapolation
is performed. Rebuild the locator when rest geometry changes. High-order curved
geometry is not accepted; its conservative bounds require a separate implementation.
This API locates points only; it does not construct surface displacement weights.

Build the global point-displacement matrix from the embedding and a displacement
basis (which may differ from the locator's geometry basis):

```python
W = fem.build_point_displacement_map[fem.HexHermiteBasis](embedding, u_dof_map, X_fields)
kinematics = dof.SurfaceKinematics(surface.positions, W)
```

C++ uses `fem::build_point_displacement_map<UBasis>(embedding, u_dof_map, X_fields)`
from `litepgo/fem/point_displacement_map.hpp`. Supported displacement bases are
TetP1, HexQ1 and HexHermite. The result has `3 * number_of_points` rows in xyz order
and `u_dof_map.num_dofs` columns (Python: SciPy CSC). Fields and the DOF map must
use the same element order and rest geometry as the embedding. Each element's
`BasisDofTransform` is built once; interpolation rows are pulled back to physical
DOFs before assembly. Hermite retains the positive axis-aligned box restriction.
Invalid IDs, nonfinite/out-of-domain xi, incompatible field counts and DOF layouts
are rejected; empty point sets produce a zero-row matrix.

Python uses the same exact boundary path:

```python
extracted = mesh.extract_surface_with_vertices(volume)
W = dof.build_vertex_displacement_map(extracted.vertices, num_dofs, dofs_per_vertex=3)
kinematics = dof.SurfaceKinematics(extracted.mesh.positions, W)
```

Python exposes the general spatial index separately from `contact.SpatialHash`:

```python
from litepgo import spatial
boxes = [spatial.AABB([0, 0, 0], [1, 1, 1]), spatial.AABB([.5, .5, .5], [2, 2, 2])]
index = spatial.SpatialHash(boxes, cell_size=1)
ids = index.query(spatial.AABB([.75, .75, .75], [.75, .75, .75]))
pairs = index.overlapping_pairs()  # Optional filter(i, j).
```

Input boxes are copied. IDs refer to their original order; output order is unspecified.
Native detection releases the GIL. Python pair filters run serially afterward, with
exceptions propagated; they do not prune native candidate enumeration.
`fem.PointEmbedding(element_ids, xi)` copies nonnegative integer IDs and finite `(n, 3)`
coordinates with matching row counts. The displacement-map builder validates element
bounds and reference-domain membership against the actual model.

Paired C++/Python examples:
- `examples/spatial/spatial_query.cpp` / `.py`: point, region and filtered pair queries.
- `examples/fem/embedding/point_embedding.cpp` / `.py`: locator, explicit embedding, W and direct
  interpolation comparisons for TetP1, HexQ1 and HexHermite (Q1 rest geometry).

Build with `cmake --build build/release --target spatial_query point_embedding`.
Run the executables under `build/release/examples/spatial/` and `build/release/examples/fem/embedding/`, or the Python files
with the configured environment. Both include numerical self-checks. The contact
example retains its exact extracted-vertex path; these examples do not sample
Hermite boundary surfaces.

Uniform Hex Q1 boundary sampling is available independently of the displacement basis:

```python
sampled = fem.sample_boundary[fem.HexQ1Basis](topology, X_fields, subdivisions=4)
W = fem.build_point_displacement_map[fem.HexHermiteBasis](
    sampled.embedding, u_dof_map, X_fields
)
kinematics = dof.SurfaceKinematics(sampled.mesh.positions, W)
```

C++ uses `fem::sample_boundary<fem::HexQ1Basis>(topology, X_fields, subdivisions)`
from `litepgo/fem/sample_boundary.hpp`. `SampledBoundary` contains `mesh` and
`embedding`. Each boundary quad produces `2 * subdivisions^2` triangles. The
reference coordinates are generated directly, without locator inversion. Shared
edge/corner samples are welded using topology and integer interpolation weights,
not coordinate proximity. Original boundary vertices appear first in increasing
volume-vertex order; subdivision 1 exactly matches `extract_surface` connectivity,
winding and vertex order. Hosts are chosen from the first boundary face encountered
in element order. Adjacent fields must agree exactly at shared corner positions,
and geometry must be conforming with positively oriented cells, as for extraction.
The sampler does not certify global element injectivity. Invalid subdivisions,
field counts, shared-corner disagreement, nonmanifold faces (more than two cells),
and index overflow are rejected.

Only Hex Q1 rest geometry is instantiated. Hermite displacement remains restricted
to axis-aligned boxes by `BasisDofTransform`; it contributes derivative weights at
edge/face samples, but not at original corners. The triangles approximate the
curved deformed boundary; sampling adds contact vertices, not simulation DOFs.
The existing Tet P1 contact examples continue using exact extracted vertex IDs.
