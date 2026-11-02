# litepgo

litepgo is a minimal adaptation of libpgo for geometry processing, finite
element simulation, contact, and optimization. It provides a modular C++20
library and Python bindings built with nanobind.

## What is included

- Mesh data structures, OBJ/VEG I/O, surface extraction, and spatial queries.
- Volumetric and shell FEM, including prescribed plastic fields and point embedding.
- Scalar energies, vector responses, Newton and Gauss–Newton solvers.
- Native IPC contact, with an optional IPC Toolkit backend.
- Paired C++ and Python examples for simulation, fitting, and geometry processing.

## Build and test

Install Python 3.12, [uv](https://docs.astral.sh/uv/), CMake 3.28+, Ninja, and a
C++20 compiler. From the repository root:

```sh
uv sync --locked
cmake --workflow --preset debug
```

After building, inspect the Python package and its configured backends:

```sh
uv run --no-sync python -c "import litepgo; print(litepgo.build_info())"
```

## Examples

CMake targets use the source filename as their name. For example, after the
`debug` workflow above:

```sh
cmake --build build/debug --target custom_energy
build/debug/examples/energy/scalar/custom_energy
uv run --no-sync python examples/energy/scalar/custom_energy.py
```

The [example catalog](docs/examples.md) covers energy, solvers, FEM, contact,
mesh processing, and spatial queries.

## Documentation

The documentation covers [build and development](docs/guide/build.md),
[CMake modules](docs/guide/development.md), [mesh I/O](docs/api/mesh.md),
[FEM](docs/api/fem.md), [contact](docs/api/contact.md), and
[spatial queries and embedding](docs/api/spatial.md).

The `uv` dependency group `docs` installs Node.js 24.19.0 and Corepack through
[`nodejs-wheel`](https://pypi.org/project/nodejs-wheel/), Doxygen 1.18.0 through
[`mlc-doxygen-bin`](https://pypi.org/project/mlc-doxygen-bin/), and pdoc. These
versions and artifacts are locked in `uv.lock`. Corepack selects pnpm 11.25.0
from `docs/package.json`; pnpm installs the VitePress dependencies from
`docs/pnpm-lock.yaml`. No global Node.js, pnpm, or Doxygen installation is needed.
The documentation launcher installs the locked dependencies automatically.
The first Corepack invocation downloads the pinned pnpm version to its cache.
The Python API generator imports the compiled extension, so build that first.
From the repository root:

```sh
uv sync --locked
cmake --preset release
cmake --build --preset release --target _core
uv run scripts/docs.py
```

The default command is `dev`: it generates both API references and starts the
local site.
To build and preview the complete site:

```sh
uv run scripts/docs.py build
uv run scripts/docs.py preview
```

Doxygen reads the public C++ headers, excluding bindings and internal `detail/`
directories, and writes to the ignored `docs/public/reference/cpp/`. The site
includes those pages under **C++ reference**. After editing headers, restart
`dev` or rerun the build to regenerate them.

Most header comments currently use ordinary `//` syntax, so the generated
reference primarily provides declarations and source browsing. The handwritten
pages explain API behavior and Python usage.

The **Python reference** documents the public modules, nanobind signatures and
docstrings, and compiled FEM template instances. Its contents reflect the
active build, including platform-specific sparse solvers and optional backends.
After changing bindings, rebuild `_core`, then restart `dev` or rerun the
documentation build. `PYTHONPATH=build/<preset>/python` selects a
particular configured build.

The optional `litepgo.torch` reference is generated when PyTorch is installed;
otherwise the generator reports that it was skipped. Install it with
`uv pip install torch` if you need the adapter's API documentation.
