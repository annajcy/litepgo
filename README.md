# litepgo

A simplified version of [libpgo](https://github.com/bohanwang/libpgo).

- **Geometry:** triangle/tet/hex meshes, OBJ and VEG I/O, surface extraction,
  spatial queries, and point embedding.
- **Simulation:** Neo-Hookean solids, Koiter StVK shells, prescribed plastic
  fields, and native IPC contact.
- **Optimization:** sparse energy assembly, constrained Newton, damped
  Gauss–Newton, and analytic equilibrium sensitivities.
- **Examples:** 26 matching C++/Python programs and four optional PyTorch fitting
  programs, with formulations and API walkthroughs in the documentation.

## Build and verify

Install Python 3.12, [uv](https://docs.astral.sh/uv/), CMake 3.28+, Ninja, and a
C++20 compiler. On Windows use an x64 Visual Studio developer shell. From this checkout:

```sh
uv sync --locked
cmake --workflow --preset release
uv run --no-sync python -c "import litepgo; print(litepgo.build_info())"
```

The workflow configures, builds, and runs C++ and Python tests. CMake fetches
native dependencies and builds the Python extension; `uv sync` installs Python
packages. See [build and environment setup](docs/guide/build.md) for individual
steps, presets, optional backends, and selecting a Python build.

## Run an example

After configuring the release build:

```sh
cmake --build build/release --target custom_energy litepgo_python
build/release/examples/energy/scalar/custom_energy
uv run --no-sync python examples/energy/scalar/custom_energy.py
```

Both programs check the same scalar-energy derivatives. Start with the
[first complete solve](docs/guide/quickstart.md), or choose a task from the
[example catalog](docs/examples.md).

## Read and build the documentation

Read the [online documentation](https://annajcy.github.io/litepgo/), or browse the
[documentation sources](docs/index.md) for the reading order. The
[architecture guide](docs/guide/architecture.md) maps the system and its modules.
The manual explains data conventions, mathematical models, and API composition; each example
page includes its formulation, run commands, expected results, and complete source.

Build and register the Python extension before generating API documentation:

```sh
uv sync --locked
cmake --preset release
cmake --build build/release --target litepgo_python
uv run scripts/docs.py          # generate both API references and serve the site
```

```sh
uv run scripts/docs.py build    # write docs/.vitepress/dist/
uv run scripts/docs.py preview  # serve the existing build
```

The launcher installs locked documentation dependencies automatically. The `docs`
uv group supplies Node.js/Corepack, Doxygen, and pdoc; Corepack selects the pinned
pnpm version and pnpm installs VitePress, MathJax, and Mermaid from its lockfile. A global
Node.js, pnpm, or Doxygen installation is unnecessary. Initial setup needs network
access. Generated pages and dependencies are ignored by Git.

Python reference pages describe the active compiled build. Set
`PYTHONPATH=build/<preset>/python` before running the launcher to select one.
PyTorch is optional: `uv pip install torch` enables the Torch examples and their
API reference. See [development](docs/development.md) for source organization,
tests, bindings, and documentation maintenance.

GitHub Actions builds the manual and both API references on pull requests and
deploys successful `main` builds to GitHub Pages.
