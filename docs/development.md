---
prev:
  text: API references
  link: /reference
---

# Development and documentation maintenance

The library is built from explicit CMake modules. Source code, tests, examples,
and documentation share one checkout so behavior and explanations can change
together. Run commands below from its root.

## Source layout and CMake dependencies

| Directory | Role |
| --- | --- |
| `src/core/` | C++ public modules and their implementations |
| `src/binding/` | nanobind registrations |
| `src/python/` | Public Python facades and Torch adapters |
| `examples/` | Standalone paired programs and Python-only Torch examples |
| `tests/cpp/`, `tests/python/` | Unit and integration/regression tests |
| `docs/` | User manual, formulations, example explanations, and site configuration |
| `assets/` | Bundled test/example geometry and provenance notes |
| `scripts/` | Development registration, benchmarks, and documentation launchers |

The source directories do not contain an extra `litepgo/` level. CMake creates
forwarding headers under `build/<preset>/include/litepgo/`, preserving public
`<litepgo/...>` includes. The importable Python package is assembled under
`build/<preset>/python/litepgo/` from `src/python/` and the compiled extension.

Link the targets your C++ source uses. Public dependencies propagate includes,
C++20 requirements, and required libraries.

| Target | Direct litepgo dependencies |
| --- | --- |
| `litepgo::common` | None; fundamental types and build information |
| `litepgo::spatial` | common |
| `litepgo::linalg` | common |
| `litepgo::linalg_accelerate` | linalg; enabled with `LITEPGO_WITH_ACCELERATE` on macOS |
| `litepgo::linalg_mkl` | linalg; enabled with `LITEPGO_WITH_MKL` on x86-64 Linux/Windows |
| `litepgo::mesh` | common |
| `litepgo::dof` | common, linalg, mesh |
| `litepgo::energy` | common, dof, linalg |
| `litepgo::fem` | common, dof, energy, mesh, spatial |
| `litepgo::solver` | common, dof, energy, linalg, configured sparse backend |
| `litepgo::contact` | common, energy, mesh, spatial |
| `litepgo::mesh_gmsh` | mesh; optional |
| `litepgo::contact_ipctk` | contact; optional |

`litepgo::core` is an INTERFACE aggregate of the nine base modules, including
OBJ/VEG I/O through `mesh` and the configured sparse backend through `solver`.
The base `linalg` target supplies Eigen solvers; Accelerate and oneMKL are separate
targets. Gmsh I/O and the IPC Toolkit backend are optional
extensions. There is no single `liblitepgo.a` to link
manually. The module dependency graph is checked for cycles.

`litepgo_binding` builds the native extension, whose Python module name remains
`litepgo._core`. `litepgo_python` builds that extension and assembles/registers
the Python package. Use `litepgo::core` as the aggregate C++ link target, or link
the individual module targets listed above.

For a separate small application, a minimal CMake file is:

```cmake
cmake_minimum_required(VERSION 3.28)
project(my_app LANGUAGES CXX)
# Replace this absolute path with the litepgo source checkout on your machine.
add_subdirectory(/absolute/path/to/litepgo litepgo)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE litepgo::fem litepgo::solver litepgo::mesh)
```

Use `#include <litepgo/...hpp>` in `main.cpp`. Domain C++ namespaces are
`litepgo::dof`, `energy`, `fem`, `linalg`, `mesh`, `solver`, `contact`, and
`spatial`. `common/` is infrastructure: basic `Vec`, `SpMat`, `BuildInfo`, and
`build_info()` remain in `litepgo`. Python bindings fill matching submodules.

## Build and test a change

```sh
cmake --preset release
cmake --build build/release --target litepgo_tests litepgo_python
ctest --test-dir build/release --output-on-failure
```

`litepgo_tests` is a build target, not an executable. Native executables are
`litepgo_<module>_tests` plus `litepgo_examples_tests`. CTest also registers
standalone regression examples and pytest. For a focused check:

```sh
ctest --test-dir build/release -R '^FemFit\.' --output-on-failure
PYTHONPATH=build/release/python uv run --no-sync python -m pytest -q tests/python/test_api_examples.py
```

Run checks appropriate to the changed behavior: finite differences for analytic
derivatives, independent formulation checks for energies, and C++/Python agreement
for paired examples. Do not replace a physical invariant with a snapshot of one
platform's final digits. Platform sparse factorizations can round differently.

On Unix, configuring a preset points the root `compile_commands.json` at that
build for clangd. The latest configured ordinary preset wins; ASan leaves the
link alone. On Windows point clangd at `build/<preset>/compile_commands.json`.
C++ headers use `.hpp`. To apply the pinned formatter:

```sh
git ls-files '*.hpp' '*.cpp' | xargs uv run --no-sync clang-format -i
```

## Add or extend a Python binding

Register native classes/functions in `src/binding/<module>/`, then expose
public aliases through the corresponding facade. Maintain its `__all__` list.
For FEM templates, register the concrete instantiation and its argument tuple;
Python `Template` objects dispatch only to compiled instances.

Validate dimensions, finite values, ownership, sparse structure, and exceptional
states at binding boundaries. Update caller examples and independent regression
tests. Build `litepgo_python` after adding/removing modules or changing native code;
it rebuilds `_core` and refreshes source links through `scripts/litepgo_dev.py`.

## Build and maintain this documentation

```sh
uv run scripts/docs.py          # dev server
uv run scripts/docs.py build    # generated references plus static site
uv run scripts/docs.py preview  # existing static build
```

The launcher synchronizes locked site packages. The uv `docs` group supplies
Node.js/Corepack, Doxygen, and pdoc. Corepack reads the pnpm version from
`docs/package.json`; pnpm reads `docs/pnpm-lock.yaml`. The first setup downloads
tools and packages. `_core` must already be importable for Python API generation.
Use `PYTHONPATH=build/<preset>/python` to select the package being documented.

Doxygen reads public headers and excludes bindings/internal detail headers. pdoc
reads public Python modules and explicitly includes bound FEM template instances.
PyTorch is a locked default Python dependency, and pdoc always includes
`litepgo.torch` alongside the native API facades.
Generated `docs/public/reference/` and `.vitepress/dist/` are ignored.

Keep one authoritative explanation per topic in `docs/`. The source under
`examples/` is included into example pages at build time, avoiding copied code.
Keep each module's overview and architecture in its `index.md`, with distinct
API/workflow topics on child pages. Short modules can remain a single page.
Write formulations with LaTeX `$...$` and `$$...$$`; the site renders them with
MathJax. Use fenced `mermaid` blocks for architecture and numerical-flow diagrams;
the documentation theme renders them in the site. Add `accTitle` and
`accDescr` to describe each diagram, and explain whether arrows mean dependencies
or data flow. System and module diagrams belong in the User guide.
Define symbols and derivative/packing conventions before using them.
When changing behavior, update the relevant manual and example narrative together.

Validation includes a site build, the documented runnable code, and real browser
clicks on generated references. API HTML links must use full HTML navigation;
VitePress's document router cannot load Doxygen/pdoc pages as Markdown routes.

The Markdown integration check runs with:

```sh
uv run --locked --group docs --directory docs corepack pnpm exec node --test .vitepress/mermaid.test.mjs
```

After changing Mermaid or its theme component, also inspect the architecture
page's two diagrams, light/dark switching, and navigation between module pages.
A static site build does not execute client-side diagram rendering.

## GitHub Pages

The `Documentation` workflow builds the Python extension, generates the C++ and
Python references, checks Markdown rendering, and uploads the complete static
site. Pull requests only build; pushes to `main` deploy to
[GitHub Pages](https://annajcy.github.io/litepgo/). Manual runs deploy only when
the selected branch is `main`.

Set **Settings → Pages → Build and deployment → Source** to **GitHub Actions**
when setting up the repository. Deployment uses the `github-pages` environment
and GitHub's built-in token; no personal token or separate publishing branch is
needed.

CI sets `DOCS_BASE=/litepgo/` for the project URL. To check the same layout locally:

```sh
DOCS_BASE=/litepgo/ uv run scripts/docs.py build
DOCS_BASE=/litepgo/ uv run scripts/docs.py preview
```

Open the preview's `/litepgo/` path. The site uses explicit `.html` links so pages
work on static hosts without URL rewriting. API links and their return links
must retain the same project prefix.
