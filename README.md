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

## Formatting

C++ headers use the `.hpp` extension.

```bash
git ls-files '*.h' '*.hpp' '*.cpp' | xargs uv run --no-sync clang-format -i
```

## Mesh I/O

`litepgo::io` (CMake target) provides `litepgo::io::read_obj/write_obj`
and `read_veg/write_veg`. Include `litepgo/io/obj.hpp` or `veg.hpp`.
`MeshData<K>` owns vertex-major xyz doubles and a `MeshTopology<K>`;
`TriangleMesh`, `TetMesh`, `HexMesh` name the supported shapes.
VEG readers return `VolumeMesh`, a tet/hex variant.

Enable `-DLITEPGO_WITH_GMSH=ON` for the separate `litepgo::io_gmsh`
target and `litepgo/io/msh.hpp`. CMake FetchContent downloads and builds pinned Gmsh 4.13.1 sources.
Tinyobjloader v2.0.0rc13 is also fetched and built with double precision.
The default build does not require Gmsh. MSH uses Gmsh for input and writes
ASCII MSH 4.1. Only homogeneous tet4 or hex8 volume meshes are accepted;
unsupported/mixed volume cells and empty volume meshes are rejected.
The backend owns a serialized Gmsh session and rejects an already initialized
external session rather than destroying its state.

Python exposes the same operations through `from litepgo import io`:

```python
mesh = io.read_veg("model.veg")
assert mesh.positions.dtype.name == "float64"
io.write_veg("copy.veg", mesh)
mesh = io.TetMesh(mesh.positions, mesh.cells)
# io.has_gmsh reports whether read_msh/write_msh are enabled.
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

Run `tests/python/test_io.py` for precision, real-asset roundtrip, sparse MSH tags,
and invalid-input checks. `scripts/check_mesh_io_reference.py` additionally compares
against an installed `pypgo`: OBJ/VEG in both directions, and MSH output through
libpgo's tet reader. Run it with both packages importable in the same Python process.
