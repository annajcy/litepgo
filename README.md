# litepgo

Minimal adaptation of libpgo.

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

```bash
git ls-files '*.h' '*.hpp' '*.cpp' | xargs uv run --no-sync clang-format -i
```
