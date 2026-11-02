# Build and environment setup

This page builds the C++ library, the Python extension, and the regression suite
from a source checkout. CI tests Debug and Release builds on macOS, Linux, and
Windows. CMake configuration may download native dependencies on the first run.

## Prerequisites and first build

Install Python **3.12**, uv, **CMake 3.28 or newer**, Ninja, and a C++20 compiler.
On macOS use the Xcode command-line tools; on Linux use a suitable GCC or Clang.
On Windows install Visual Studio Build Tools with the C++ workload and run these
commands from an x64 Native Tools Command Prompt or Developer PowerShell.
From the repository root:

```sh
uv sync --locked
cmake --workflow --preset release
uv run --no-sync python -c "import litepgo; print(litepgo.build_info())"
```

`uv sync` creates `.venv` and installs NumPy, SciPy, pytest, and the development
tools. It does not build litepgo. The CMake workflow performs configuration,
compilation, and CTest; CTest runs both native tests and pytest. A nonzero exit
means a stage failed. `build_info()` reports the compiled versions and available
backends rather than assuming that an optional feature exists.

For just the Python extension, without running the full test workflow:

```sh
cmake --preset release
cmake --build build/release --target litepgo_python
```

## Configure, build, and test separately

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
```

| Preset | Configuration | Python extension |
| --- | --- | --- |
| `release` | Optimized build | Yes |
| `debug` | Debug build | Yes |
| `relwithdebinfo` | Optimized build with debug symbols | Yes |
| `asan` | Address/undefined-behavior sanitizers | No |

The workflow presets are `release`, `debug`, and `asan`. Configure/build/test
presets also exist for `relwithdebinfo`. Build directories are
`build/<preset>/`. Keep machine-specific overrides in the ignored
`CMakeUserPresets.json`.

Examples are CMake targets named after their source stem and are excluded from
the default build unless a test depends on them. Build a particular example
explicitly:

```sh
cmake --build build/release --target newton_chain
build/release/examples/solver/newton_chain
```

On Windows append `.exe` to compiled example paths. The build copies the TBB DLL
beside executables and the Python extension, so no extra runtime `PATH` setup is
required.

## How Python finds the extension

The default build, or the explicit `litepgo_python` target, creates
`build/<preset>/python/litepgo/`, containing `_core` and links to the Python
facade files. It registers that directory in `.venv` through `litepgo_dev.pth`.
The most recently built registration is the default. CMake selects the active
`VIRTUAL_ENV`, or the checkout's `.venv`.

To select a build explicitly, including in subprocesses:

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/solver/newton_chain.py
PYTHONPATH=build/release/python uv run --no-sync python -m pytest -q tests/python/test_api_examples.py
```

In PowerShell set `$env:PYTHONPATH = "build/release/python"` before the command
instead of using a shell-variable prefix. Building `litepgo_python` selects the
default package without needing `PYTHONPATH`.

Editing an existing facade `.py` file takes effect immediately through the links
on Unix. Windows uses copies to avoid requiring symlink privileges; rebuild
`litepgo_python` after editing a facade there.
Rebuild `litepgo_binding` after changing C++. Adding/removing a Python module requires
building `litepgo_python` to refresh the source links and environment registration.
An ASan-only build cannot provide an importable extension for ordinary Python.

## Sparse backends and optional features

Eigen LDLT and SparseLU are always available. `AutoSparseSolver` selects the
platform's positive-definite backend: Accelerate on macOS, oneMKL/PARDISO on
x86-64 Linux when enabled, otherwise Eigen LDLT. `AutoIndefiniteSparseSolver`
uses Accelerate LDLT, PARDISO LDLT, or Eigen SparseLU, respectively.
The default Windows build uses the Eigen backends.
See [linear algebra](linalg/index.md) for backend selection and
[solvers](solver/index.md) for nonlinear algorithms.

| CMake setting | Behavior |
| --- | --- |
| `LITEPGO_WITH_ACCELERATE` | Defaults on for macOS; system Accelerate framework |
| `LITEPGO_WITH_MKL` | Defaults on for x86-64 Linux; pinned oneMKL wheels are cached in `build/_cache/` |
| `LITEPGO_MKL_ROOT` | Use a local compatible oneMKL installation instead of downloading wheels |
| `LITEPGO_WITH_IPC_TOOLKIT` | Defaults on; adds the optional `litepgo::ipctk` backend |
| `LITEPGO_WITH_GMSH` | Defaults off; adds MSH I/O through fetched Gmsh sources |
| `LITEPGO_BUILD_PYTHON` | Enables the nanobind extension; enabled by ordinary presets |
| `LITEPGO_BUILD_TESTS` | Builds and registers the native/Python regression suites |

For example, disable the optional IPC Toolkit backend while retaining native IPC:

```sh
cmake --preset release -DLITEPGO_WITH_IPC_TOOLKIT=OFF
cmake --build --preset release
```

CMake pins and fetches Eigen, oneTBB, nanobind and other native dependencies.
oneTBB is shared. MKL uses litepgo's TBB threading runtime; TBB controls bound
its parallel work, while `MKL_NUM_THREADS` does not select that threading layer's
parallelism. Cache locations and platform-specific paths belong in local presets.

## Common setup failures

- **`import litepgo` fails:** run `uv sync --locked`, configure a Python-enabled
  preset, and build `litepgo_python`. Confirm the interpreter is the checkout's `.venv`.
- **An old API appears:** set `PYTHONPATH` to the intended build and rebuild `litepgo_binding`.
- **An example executable is missing:** build its named CMake target explicitly.
- **An optional symbol is absent:** inspect `build_info()` and the module's
  capability flag, such as `mesh.has_gmsh`; reconfigure and rebuild if needed.
- **Initial configuration is slow:** dependency downloads/builds happen before
  litepgo compilation. Later configurations reuse the caches.

Next, [solve a complete constrained problem](quickstart.md).
