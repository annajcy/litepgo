# litepgo

Minimal adaptation of libpgo.

FEM examples and the mathematical naming and derivative-layout conventions are in [example/fem](example/fem/README.md).

Requires Python 3.12, uv, CMake 3.28+, Ninja, and a C++20 compiler.

Eigen 5.0.1, oneTBB 2023.1.0, nanobind 3.1.0, and GoogleTest are fetched at pinned versions; TBB is built as a shared library. Eigen's LDLT and SparseLU are always built; the platform's sparse backends become the `AutoSparseSolver` and `AutoIndefiniteSparseSolver` defaults. `litepgo.build_info()` reports what a build contains.

- macOS: Apple Accelerate's LLT and LDLT (`LITEPGO_WITH_ACCELERATE`, on by default).
- x86-64 Linux: Intel oneMKL 2026.1 PARDISO's LLT and pivoted LDLT (`LITEPGO_WITH_MKL`, on by default). Configuring downloads the pinned `mkl` and `mkl-include` wheels from PyPI (about 225 MB) and unpacks them once into `build/_cache/` (`LITEPGO_MKL_CACHE_DIR`), shared by every preset; `-DLITEPGO_MKL_ROOT=/opt/intel/oneapi/mkl/2026.1` uses a local oneMKL 2026.1 install instead. MKL runs on its oneTBB threading layer over litepgo's own TBB runtime, so `tbb::task_arena` and `tbb::global_control` bound PARDISO like every other parallel loop; `MKL_NUM_THREADS` has no effect. Eigen's dense code does not use MKL.

Native IPC contact is always built into `litepgo::litepgo`, with no IPC Toolkit dependency.
The optional `litepgo::ipctk` target (`LITEPGO_WITH_IPC_TOOLKIT`, on by default) retains the IPC Toolkit 1.6.1 backend.
It reuses litepgo's Eigen and TBB and fetches its dependencies into `build/_cache/cpm/` (`LITEPGO_CPM_CACHE_DIR`).
`-DLITEPGO_WITH_IPC_TOOLKIT=OFF` skips the toolkit and its dependencies; native C++ and Python contact remain available.

## Build and test

```bash
uv sync --locked
cmake --workflow --preset debug
```

`uv sync` only creates `.venv` with the Python dependencies; it never builds litepgo. CMake builds the C++ library, the tests, and the nanobind extension. The extension is built against the active virtual environment (`$VIRTUAL_ENV`), else `.venv`.

Presets: `debug`, `release`, `relwithdebinfo`, `asan`. Each builds in `build/<preset>/` and runs GoogleTest and pytest through CTest. `asan` skips the extension, since a sanitized module cannot load into a regular Python. Keep local overrides in the ignored `CMakeUserPresets.json`.

## Using litepgo from Python

Every build puts an importable package in `build/<preset>/python/litepgo/`: the compiled `_core`, plus symlinks to the sources in `python/litepgo/`. `scripts/litepgo_dev.py` then writes `litepgo_dev.pth` into the venv, so `import litepgo` loads the most recently built preset:

```bash
cmake --build --preset release
uv run --no-sync python -c "import litepgo; print(litepgo.build_info())"
```

Edits to existing `.py` files take effect without a rebuild; adding or removing a module, or changing C++, needs `cmake --build`.

## clangd

Configuring a preset links `compile_commands.json` at the repository root to that build's database (the last preset configured wins). It covers the core, tests, and bindings. `asan` leaves the link alone, since it has no bindings.

## Namespaces

Each directory under `cpp/litepgo` is a module with its own namespace:
`litepgo::dof`, `energy`, `fem`, `linalg`, `mesh`, `solver`. Mesh file I/O
(`mesh/io`) is part of `litepgo::mesh`. `core`
(`Vec`, `SpMat`, ...) stays in the root `litepgo` namespace, so every module
sees it unqualified; names from other modules are written `module::Name`.
Python bindings for module `m` live in `litepgo::m::binding` and fill the
Python submodule `litepgo.m`; C++ tests for it are in `litepgo::m::test` (core:
`litepgo::test`; example fixtures: `litepgo::example::test`).

## Formatting

C++ headers use the `.hpp` extension.

```bash
git ls-files '*.h' '*.hpp' '*.cpp' | xargs uv run --no-sync clang-format -i
```

## Mesh I/O

The `litepgo::io` CMake target provides `litepgo::mesh::read_obj/write_obj`
and `read_veg/write_veg`. Include `litepgo/mesh/io/obj.hpp` or `veg.hpp`.
The core library's `litepgo/mesh/extract_surface.hpp` provides `litepgo::mesh::extract_surface(mesh)`:
the boundary of a conforming tet or hex mesh as outward-wound triangles (hex quads are split), keeping only
boundary vertices in their original order. Write it with `write_obj` to view a
volume mesh as a surface.
`MeshData<K>` owns vertex-major xyz doubles and a `MeshTopology<K>`;
`TriangleMesh`, `TetMesh`, `HexMesh` name the supported shapes.
VEG readers return `VolumeMesh`, a tet/hex variant.

Enable `-DLITEPGO_WITH_GMSH=ON` for the separate `litepgo::io_gmsh`
target and `litepgo/mesh/io/msh.hpp`. CMake FetchContent downloads and builds pinned Gmsh 4.13.1 sources.
Tinyobjloader v2.0.0rc13 is also fetched and built with double precision.
The default build does not require Gmsh. MSH uses Gmsh for input and writes
ASCII MSH 4.1. Only homogeneous tet4 or hex8 volume meshes are accepted;
unsupported/mixed volume cells and empty volume meshes are rejected.
The backend owns a serialized Gmsh session and rejects an already initialized
external session rather than destroying its state.

Python exposes the same operations, with the topologies, through `from litepgo import mesh`:

```python
data = mesh.read_veg("model.veg")
assert data.positions.dtype.name == "float64"
mesh.write_veg("copy.veg", data)
mesh.write_obj("surface.obj", mesh.extract_surface(data))
data = mesh.TetMesh(data.positions, data.topology)
# mesh.has_gmsh reports whether read_msh/write_msh are enabled.
```

Python positions have shape `(n, 3)` and cells `(m, 3/4/8)`; properties
return independent copies. Construct a new mesh to change geometry.
Readers/writers preserve geometry, not materials, UVs, normals, physical groups,
or boundary labels. OBJ polygons are triangulated by tinyobjloader; its default
triangulation is not guaranteed for arbitrary complex polygons. Supply triangulated
OBJ for simulation. OBJ decimal parsing can introduce double-level rounding.
VEG supports text TET/CUBIC and relative `*INCLUDE`, with cycle detection.
No welding, rescaling, orientation repair, or mesh generation is performed.
FEM performs its own geometric validity checks. Writes replace the named file;
invalid mesh data is rejected before opening it, and stream errors are reported.

Run `tests/python/test_mesh.py` for precision, real-asset roundtrip, sparse MSH tags,
and invalid-input checks.

## FEM from Python

`litepgo.fem` mirrors the C++ API: the same classes, constructor arguments and
template arguments. A C++ class template is a `fem.Template`; indexing it with the
template arguments, in C++ order, returns the bound instantiation. Basis, quadrature
and material classes serve as type arguments and also expose their static
evaluations (`eval_N`, `points()`, `eval_W`, ...).

```python
import numpy as np
from litepgo import dof, energy, fem, mesh, solver

data = mesh.read_veg("assets/box-hex.veg")
n = data.topology.num_cells
rest_map = dof.VertexDofMap(data.topology, 3)
XField = fem.InterpolatedField3D[fem.HexQ1Basis]
X = [XField(rest_map.gather(e, data.positions.ravel())) for e in range(n)]
Fp = [fem.IdentityField3D(np.eye(3).ravel())] * n
theta = [fem.IdentityField3D(np.array([2.0, 3.0]))] * n
Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.HexHermiteBasis, fem.GaussLegendre3D[4]]
elastic = Energy(dof.VertexDofMap(data.topology, 24), X, Fp, theta)
```

The building blocks are bound too: `LocalEnergy2D/3D`,
`VolumetricMaterialLocalEnergy{F,Fp,Theta}3D[M]` (which borrow their fields and keep
them alive), `ShellMaterialLocalEnergy2D[M]`, `VolumetricElement3D[Q]`,
`BasisDofTransform[B]` (physical DOFs to basis coefficients, e.g. Hermite slots), and the
kinematics and chain-rule functions. Where C++ fills optional output pointers or adds
into caller buffers (`eval`, `accumulate`, `eval_triangle_mid_edge_kinematics`), Python
takes `gradient=`/`hessian=` flags and returns `(value, gradient or None, hessian or None)`.
Matrix and vector arguments must be NumPy arrays.

The energies' field template arguments are erased (`fem::SharedField3D`): each
list may hold any `Field3D`, chosen per element, and the energy shares ownership
of them. Material, basis and quadrature arguments are compiled in;
`cpp/litepgo/bindings/fem_energy.cpp` lists the available instantiations, and
indexing any other combination raises `TypeError` naming them. Add one line there
to bind another. Fields and energies cannot be subclassed in Python: they are
evaluated per quadrature point inside parallel C++ loops. `example/fem/*.py` mirror
the four C++ FEM examples line by line, and `tests/python/test_fem.py` checks that
each pair writes byte-identical files. `solver.newton_solve(energy, fixed, x0, options,
solver=None)` likewise follows the C++ signature (`fixed` is a `dof.DofReduction`;
the solver defaults to `AutoSparseSolver()`), and `solver.NewtonOptions()` starts from
the C++ defaults, with a NaN tolerance that must be set before solving.

## Contact

`litepgo/contact/ipc_contact_energy.hpp` provides independent native `contact::IPCContactEnergy` and
`contact::IPCContactOptions`. `litepgo/contact/ipctk_contact_energy.hpp` provides the retained toolkit backend,
`contact::IPCTKContactEnergy` and `contact::IPCTKContactOptions`, linked with `litepgo::ipctk`.
Both use `contact::Plane`, and are `energy::Energy` implementations over vertex-major xyz surface positions
with unit stiffness. Set the stiffness as a weight in `WeightedEnergySum`. Call `update_pattern(x)` before
Hessian assembly; the Newton solver handles the dynamic sparsity pattern automatically.

Native contact implements the original squared-distance log barrier
`b(s,h) = -(s-h)^2 log(s/h)` for `0 < s < h`, zero for `s >= h`, with
`s = distance² - min_distance²` and `h = dhat * (dhat + 2*min_distance)`.
It sums nonincident vertex-triangle and unordered edge-edge candidates, with closest-feature
point-point, point-edge, point-triangle and edge-edge distances. Repeated closest-feature stencils are
merged **with their multiplicities preserved**: this is an assembly optimization, not a change from the
original candidate sum. Active-pair diagnostics count merged stencils. Edge pairs needing mollification
retain all four vertices, using `m = z*(2-z)` for `z < 1`, otherwise one, where
`z = |e0 × e1|² / (1e-3 |e0_rest|² |e1_rest|²)`. Derivatives include the distance and mollifier;
projected Hessians clamp each local term's eigenvalues. Assembly order is fixed and deterministic.
This is the unweighted original barrier, not the area-weighted/physical barrier or friction model.

The native backend uses spatial-hash AABBs, including swept boxes for linear motion, and additive
conservative advancement (ACCD). `IPCContactOptions.ccd_tolerance` is a **time increment** stopping
tolerance: when another safe increment is smaller, return the last conservative bound.
`ccd_max_iterations` bounds advancement per pair, also returning the last bound on exhaustion.
The first positive conservative increment is allowed even below the tolerance; stalled arithmetic
returns zero. Neither option switches off CCD. `IPCTKContactOptions` instead passes its tolerance and
iteration budget to Tight-Inclusion CCD. Planes use analytic signed-distance steps with a 0.8 safety factor.

Native input requires nonempty, nondegenerate triangles, unique faces, and no unused vertices.
Options, finite positions, normalized plane normals and representable geometric quantities are validated.
`eval` rejects active distances at or below `min_distance` and vertices behind planes; it is not a full
intersection query. `max_feasible_step` additionally checks initial edge-face intersections. Start solves
from feasible states. Shared-vertex primitives are excluded. Coplanar overlaps are detected through
zero closest-feature distances. Degenerate evaluated triangles are rejected.

Native distance gradients/Hessians use a minimal vendored subset of IPC Toolkit 1.6.1
analytical routines (MIT; see `cpp/litepgo/contact/detail/ipc_derivatives/README.md`
and its original LICENSE). Feature selection and scalar distances remain native.
Barrier and mollifier derivatives use explicit chain/product rules, including mixed
terms; gradient-only evaluation does not compute Hessians. This does not link the
native library to IPC Toolkit and does not change the optional IPCTK backend.

Native predicates and ACCD use double precision, without a formal floating-point robustness guarantee;
extreme scales, grazing contact and nearly degenerate geometry remain numerical limitations.
Closest-feature transitions are generally C1, not C2; exact Hessians are those of the selected branch.
Native contact accepts a shared, read-only broadphase backend as its required third argument:

```cpp
#include <litepgo/contact/broad_phase/spatial_hash.hpp>
auto broad_phase = std::make_shared<litepgo::contact::SpatialHash>();
litepgo::contact::IPCContactEnergy contact(surface, options, broad_phase);
```

```python
from litepgo.contact import IPCContactEnergy, SpatialHash
broad_phase = SpatialHash()
contact = IPCContactEnergy(surface, options, broad_phase=broad_phase)
```

The native constructor is `(surface, options, broad_phase, planes = {})`.
The backend is required: omission, `nullptr` and `None` are rejected. Explicitly pass `SpatialHash`; move any plane list to the fourth argument. Energies retain shared ownership, so one backend
can serve multiple energies and outlive the caller's reference. IPCTK's constructor is unchanged.
Python exposes the built-in base/derived types for selection, not Python-defined backend overrides.

C++ backends derive from `BroadPhase` and implement `detect(const BroadPhaseQuery&) const`.
The query contains already inflated, outward-rounded static or swept `AABB`s and a
`PairKindMask` for VF, EE and optional EF pairs. Return `BoxPair` indices into the input
array, including every allowed overlapping pair. Missing pairs violate the collision-safety
contract. Conservative extra pairs, arbitrary order and duplicates are accepted; native
contact validates indices, filters kinds/AABB overlap/topology, and sorts/uniques primitive
pairs before contact feature multiplicities are formed. Backend calls must be safe to share
concurrently; query data cannot be retained. SpatialHash validates bounds/kinds and uses per-call scratch.

`SpatialHash()` is the native backend (`SpatialHash(cell_size)` selects a
finite positive width explicitly). Auto sizing uses the median positive maximum AABB
extent among edges/faces, excluding points. It uses a sparse `unordered_map` with full
signed int3 equality; each box covers every cell from `floor(lo/h)` through `floor(hi/h)`
inclusively. Records and bucket queries are parallel; map insertion is serial. The
lowest shared cell owns each pair, preventing duplicate output without skipping overlaps.

All input boxes are validated first. Nonfinite bounds, unrepresentable int64 cell
coordinates, or more than 2^20 box-cell records throw an overflow or length error. Invalid bounds/kinds still throw. There is **no pair-work
budget or pair-work-triggered fallback**. Long overlapping AABBs may be examined in
many shared cells despite unique output; dense buckets remain quadratic. Shared
backends are safe for concurrent calls because all query scratch is local.

The benchmark compares native SpatialHash contact with IPCTK.
The examples explicitly choose SpatialHash; contact sorts merged candidates deterministically. Swept bounds, inflation and
outward rounding are preserved. It has quadratic worst-case behavior for heavily overlapping boxes. Native local contact
evaluation, Hessian projection, gradient/Hessian assembly and surface CCD queries use TBB.
Assembly owns disjoint output DOFs and sparse columns, gathering contributions in fixed pair order;
scalar energy and CCD minimum reductions remain serial. Results are bitwise deterministic across thread counts.
A per-call vertex/pair incidence list uses O(vertices + active pairs) extra memory. Broadphase box
construction/sorting and active-set narrowphase remain serial. No acceleration-cache state is retained between evaluations.
The formulation follows [IPC (2020)](https://cims.nyu.edu/gcl/papers/2020-IPC.pdf) and
[Codimensional IPC / ACCD (2021)](https://arxiv.org/abs/2012.04457).

Migration: `BarrierContactEnergy` → `IPCTKContactEnergy`, `BarrierContactOptions` → `IPCTKContactOptions`,
`barrier_contact_energy.hpp` → `ipctk_contact_energy.hpp`, `litepgo::contact` → `litepgo::ipctk`, and
`LITEPGO_WITH_IPC` → `LITEPGO_WITH_IPC_TOOLKIT`. To use the native implementation instead, choose the
`IPCContactEnergy`/`IPCContactOptions` pair; the examples do this by default.

Migration: `SurfaceDofMap` → `SurfaceKinematics`, `surface_dof_map.hpp` → `surface_kinematics.hpp`
(C++ namespace and Python module remain `dof`; no old-name alias).
Surface kinematics accepts only `(rest_positions, W)`. For extracted boundaries,
`extract_surface_with_vertices` preserves exact volume vertex IDs; use
`dof::build_vertex_displacement_map(vertices, num_dofs, dofs_per_vertex = 3)`
(`litepgo/dof/vertex_displacement_map.hpp`) to build exact value-slot weights.
Negative/out-of-range IDs are rejected. Hermite uses width 24 and only value slots;
this does not sample curved faces. Independent surfaces use the locator path below.

`dof::SurfaceKinematics` (`litepgo/dof/surface_kinematics.hpp`) gives surface positions as an affine function of the
simulation DOFs, `p(u) = X + W u`, built from a general sparse `W`; zero rows keep static vertices at rest.
The barrier acts on `u` as `AffineMappedEnergy(barrier, map.displacement_map(), map.rest_positions())`, which
pulls gradients and Hessians back as `W^T g` and `W^T H W`. The contact examples preserve extracted vertex IDs to avoid inverse-solve roundoff
introducing tiny nonzero weights and expanding the mapped Hessian pattern.
`example/contact/tet_p1_box_stack.cpp`
(and its Python mirror) presses two tet boxes together:

```cpp
const auto extracted = mesh::extract_surface_with_vertices(stack);
const auto& surface = extracted.mesh;
const auto W = dof::build_vertex_displacement_map(extracted.vertices, num_dofs);
auto barrier = std::make_shared<contact::IPCContactEnergy>(surface, contact::IPCContactOptions{.dhat = dhat},
    std::make_shared<contact::SpatialHash>());
const dof::SurfaceKinematics kinematics(surface.positions, W);
auto contact = std::make_shared<energy::AffineMappedEnergy>(barrier, kinematics.displacement_map(),
                                                            kinematics.rest_positions());
energy::WeightedEnergySum potential({{elastic, 1}, {loads, 1}, {contact, kappa}});
```

Python mirrors both backends as `litepgo.contact`; `IPCTKContactEnergy` and its options are exported only in toolkit-enabled builds. Friction and adaptive barrier stiffness are not implemented.

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
- `example/spatial/spatial_query.cpp` / `.py`: point, region and filtered pair queries.
- `example/fem/point_embedding.cpp` / `.py`: locator, explicit embedding, W and direct
  interpolation comparisons for TetP1, HexQ1 and HexHermite (Q1 rest geometry).

Build with `cmake --build build/release --target spatial_query point_embedding`.
Run the executables under `build/release/example/{spatial,fem}/`, or the Python files
with the configured environment. Both include numerical self-checks. The contact
example retains its exact extracted-vertex path; these examples do not sample
Hermite boundary surfaces.
