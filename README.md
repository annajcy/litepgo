# litepgo

Minimal adaptation of libpgo.

FEM examples and the mathematical naming and derivative-layout conventions are in [example/fem](example/fem/README.md).

Requires Python 3.12, uv, CMake 3.28+, Ninja, and a C++20 compiler.

Eigen 5.0.1, oneTBB 2023.1.0, nanobind 3.1.0, and GoogleTest are fetched at pinned versions; TBB is built as a shared library. On macOS the Accelerate framework provides an extra sparse LDLT backend (`LITEPGO_WITH_ACCELERATE`, on by default); elsewhere only Eigen's LDLT is built. `litepgo.build_info()` reports what a build contains.

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
them alive), `ShellMaterialLocalEnergy2D[M]`, `VolumetricElement3D[Q]`, and the
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
evaluated per quadrature point inside parallel C++ loops.
