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
| `src/python/` | Public Python facades, Python-material bases and Torch adapters |
| `tutorials/torch/` | Notebook course sources; generated pages live under `docs/tutorials/torch/` |
| `examples/` | Standalone paired programs and Python-only material and Torch examples |
| `tests/cpp/`, `tests/python/` | Unit and integration/regression tests |
| `docs/` | User manual, formulations, example explanations, and site configuration |
| `assets/` | Bundled test/example geometry and provenance notes |
| `benchmarks/` | Reproducible performance drivers; see [benchmarks](benchmarks/index.md) |
| `scripts/` | Development registration and documentation launchers |

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
`litepgo_<module>_tests`. CTest also registers
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

## Performance fix: Hex Hermite EGH (2026-11-07)

A regression investigation against the October 29 FEM/libpgo benchmark found
that arbitrary-stride output views hid the actual row-contiguous batch layout
from Eigen. The change to `StridedMatRef` in `a7f7499` preserved correctness but
made the repeated Hessian contraction $B^T C B$ slower. Here $B$ is the
deformation-gradient Jacobian with respect to local coefficients, and $C$ is
the material Hessian. Hex Hermite magnifies this cost: each element has 192
displacement DOFs and the tested rule has 64 quadrature points.

The fix exposes the existing layout without copying matrices or changing the
batch protocol:

- `fem/assembly/dof_derivatives.cpp`: use a row-major `Eigen::Map` with unit
  inner stride when accumulating into row-contiguous Hessian storage.
- `fem/assembly/basis_dof_transform.hpp`: use the same layout information for
  Hermite's physical-DOF Hessian scaling and finite-value check.
- `dof/energy_assembler.cpp`: scan packed local Hessians in physical memory
  order when validating them. Padded or arbitrary-stride views retain the
  general path; unused padding must not be read.

All paths retain dimension, finite-value, overflow, and sparse-pattern checks.
The physical transform, assembly contribution order, and public APIs are unchanged.

### Measurement setup and results

Measurements used an Apple M1 Pro (6 performance and 2 efficiency cores),
AppleClang 21, Release `-O3 -DNDEBUG`, Eigen 5, the same oneTBB runtime,
`EIGEN_DONT_PARALLELIZE`, and `VECLIB_MAXIMUM_THREADS=1`.
The main case has 216 shared-vertex Hex Hermite elements (8232 global DOFs),
nonuniform axis-aligned boxes, `batch_size=256`, and a 4-by-4-by-4 Gauss rule.
The material is Neo-Hookean with lambda=2, mu=3, and identity Fp. EGH means
energy, gradient, and the **exact** Hessian together; construction and solver
iterations are excluded. Global sparse assembly is included.

The contraction experiment alternated the saved old contraction and optimized
contraction in the same process, using identical current FEM and assembly code.
Seven timed samples per case followed warmup; the table gives the final repeat's
medians. Checks also covered 64 elements, three nonzero displacement states,
and 1/4/8 threads.

| Threads | Before contraction fix | After contraction fix | Time saved |
| --- | ---: | ---: | ---: |
| 1 | 442.43 ms | 338.53 ms | 23.5% |
| 4 | 118.77 ms | 91.74 ms | 22.8% |
| 8 | 86.68 ms | 64.63 ms | 25.4% |

A separate follow-up measured the scaling and packed-validation fixes after
both binaries already included the contraction fix. Three rounds alternated
process order, with warmup and 11 samples per case. Values below are medians
of the three per-round medians; these are separate runs from the preceding table.

| Threads | Output storage | Before follow-up | After follow-up |
| --- | --- | ---: | ---: |
| 4 | Owned public result | 93.01 ms | 87.63 ms |
| 4 | Reused outputs and sparse pattern | 90.99 ms | 85.90 ms |
| 8 | Owned public result | 63.64 ms | 62.27 ms |
| 8 | Reused outputs and sparse pattern | 61.16 ms | 59.79 ms |

At four threads, scaling and its finite-value check fell from about 14.6 to
4.4 ms of **summed element CPU time**; this sum is not wall-clock time.
Assembly wall time fell from about 4.9 to 4.3 ms.

### Historical timing alignment and validation

The historical benchmark reused its gradient, Hessian, and sparse structure.
The current public API returns owned outputs and additionally validates finite
inputs and derivative results. A benchmark-only preallocated-output hook aligned
allocation and sparse-pattern reuse; it is not a new public API and does not
make the historical and current validation code identical.

The retained historical executable reran at 59.83 ms for 216 elements and eight
threads, close to the optimized reused-output path's 59.79 ms. Treat this as
agreement within run variability, not an exact cross-version speedup. Thermal
and scheduling differences remain; the alternating before/after experiments
provide stronger evidence for each optimization than comparisons across dates.
These numbers do not establish performance on other machines, projected
Hessians, other element families, or complete solvers.

Validation passed 299 C++ tests (162 FEM, 56 DOF, 81 energy) and 67 Python tests.
Regression coverage includes packed and padded row/column layouts, arbitrary
strides with non-finite padding, signed and zero weights, overflow, unchanged
outputs after invalid assembly inputs, and padded Hermite scaling against a
dense reference. A/B checks found no energy, gradient, or Hessian differences
in the tested states.

Local investigation sources, raw samples, phase measurements, and environment
metadata were saved under `build/hermite-egh-investigation/`, including the
`ab/` and `followup/` directories. These local build artifacts are not committed;
the method and measured summaries above are the durable regression record.

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
uv run docs/scripts/site.py          # dev server
uv run docs/scripts/site.py build    # generated references plus static site
uv run docs/scripts/site.py preview  # existing static build
```

For the first documentation build on macOS, Linux, or WSL, install micromamba
and prepare the build-local C++20 kernel with
`uv run scripts/setup_cpp_tutorial.py --no-register`. The site executes the
Energy and Torch notebooks from fresh kernels for each build.

The launcher synchronizes locked site packages. The uv `docs` group supplies
Node.js/Corepack, Doxygen, and pdoc. Corepack reads the pnpm version from
`docs/package.json`; pnpm reads `docs/pnpm-lock.yaml`. The first setup downloads
tools and packages. `_core` must already be importable for Python API generation.
Use `PYTHONPATH=build/<preset>/python` to select the package being documented.

`docs/scripts/cpp_reference.py` runs Doxygen on public headers, excluding
bindings and internal detail headers. `docs/scripts/python_reference.py` uses
pdoc to read public Python modules and explicitly includes bound FEM template
instances.
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
DOCS_BASE=/litepgo/ uv run docs/scripts/site.py build
DOCS_BASE=/litepgo/ uv run docs/scripts/site.py preview
```

Open the preview's `/litepgo/` path. The site uses explicit `.html` links so pages
work on static hosts without URL rewriting. API links and their return links
must retain the same project prefix.
