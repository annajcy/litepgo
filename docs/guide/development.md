# Development

## CMake modules


Link the modules used by your source files; their public dependencies propagate
include paths, C++20, Eigen/TBB settings and required libraries automatically.
Each module owns its source list in `cpp/litepgo/<module>/CMakeLists.txt`.

| Target | Type | Direct litepgo module dependencies |
| --- | --- | --- |
| `litepgo::common` | Header-only | None |
| `litepgo::spatial` | Header-only | common |
| `litepgo::linalg` | Static | common; optional Accelerate/MKL backends |
| `litepgo::mesh` | Static | common |
| `litepgo::dof` | Static | common, linalg, mesh |
| `litepgo::energy` | Static | common, dof, linalg |
| `litepgo::fem` | Static | common, dof, energy, mesh, spatial |
| `litepgo::solver` | Static | common, dof, energy, linalg |
| `litepgo::contact` | Static | common, energy, mesh, spatial |
| `litepgo::io` | Static | mesh |
| `litepgo::io_gmsh` (optional) | Static | io |
| `litepgo::ipctk` (optional) | Static | contact |

```cmake
target_link_libraries(my_fem_app PRIVATE litepgo::fem litepgo::solver litepgo::io)
# Existing consumers can keep the broad entry point:
target_link_libraries(my_existing_app PRIVATE litepgo::litepgo)
```

`litepgo::litepgo` is now an INTERFACE aggregate of the nine base modules, not a
single archive. It retains its previous scope: IO and IPC Toolkit are linked
separately. No module links back to the aggregate; configuration rejects cycles
between litepgo targets. Code that manually links `liblitepgo.a` must migrate to
CMake targets (or explicitly link the module archives).

Tests build as `litepgo_<module>_tests`, plus `litepgo_examples_tests` for shared
example fixtures. `cmake --build build/release --target litepgo_tests` still builds
all tests and their regression examples, but `litepgo_tests` is now a build target,
not an executable. Run all cases with `ctest --test-dir build/release`, or one
module executable such as `build/release/tests/cpp/litepgo_solver_tests`.
The Python extension links the modules it exposes and keeps the same Python API.


## clangd


Configuring a preset links `compile_commands.json` at the repository root to that build's database (the last preset configured wins). It covers the library, tests, and bindings. `asan` leaves the link alone, since it has no bindings.


## Namespaces


Domain modules have their own namespaces, such as `litepgo::dof`,
`litepgo::energy`, `litepgo::fem`, `litepgo::linalg`, `litepgo::mesh` and
`litepgo::solver`. Mesh file I/O (`mesh/io`) is part of `litepgo::mesh`.

`common/` groups shared infrastructure; it does not define a public
`litepgo::common` API. Fundamental types (`Vec`, `SpMat`, vector references),
`BuildInfo` and `build_info()` stay in the root `litepgo` namespace.
Internal helpers `BusyScope` and `overlaps()` live in `litepgo::detail`.
The header paths are `litepgo/common/*.hpp`, and their CMake target is
`litepgo::common`; the former `core/` paths and `litepgo::core` target are removed.

Python bindings for module `m` live in `litepgo::m::binding` and fill the
Python submodule `litepgo.m`. The common binding lives in
`litepgo::common::binding` but exposes `build_info` on the native module root;
the public Python call remains `litepgo.build_info()`.
C++ tests use `litepgo::m::test`, except common tests (`litepgo::test`) and
example fixtures (`litepgo::example::test`).


## Formatting


C++ headers use the `.hpp` extension.

```bash
git ls-files '*.h' '*.hpp' '*.cpp' | xargs uv run --no-sync clang-format -i
```
