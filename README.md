# litepgo

Minimal adaptation of libpgo.

Requires Python 3.12, uv, CMake 3.28+, Ninja, and a C++20 compiler.

## C++ build and tests

```bash
cmake --workflow --preset debug
```

Presets: `debug`, `release`, `relwithdebinfo`, `asan`. Each builds in `build/<preset>/` and runs GoogleTest through CTest. Keep local overrides in the ignored `CMakeUserPresets.json`.

## Python package

```bash
uv sync --locked
uv run --no-sync python -m pytest -q
```

`uv sync` builds the nanobind extension through scikit-build-core. `[tool.uv] cache-keys` makes it rebuild after any change under `cpp/`, `bindings/`, `cmake/`, or to `CMakeLists.txt`; add new native directories there. `uv sync --reinstall-package litepgo` forces a rebuild.

## Formatting

```bash
git ls-files '*.h' '*.hpp' '*.cpp' | xargs uv run --no-sync clang-format -i
```
