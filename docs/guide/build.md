# Build and test

Requires Python 3.12, uv, CMake 3.28+, Ninja, and a C++20 compiler.

Eigen 5.0.1, oneTBB 2023.1.0, nanobind 3.1.0, and GoogleTest are fetched at pinned versions; TBB is built as a shared library. Eigen's LDLT and SparseLU are always built; the platform's sparse backends become the `AutoSparseSolver` and `AutoIndefiniteSparseSolver` defaults. `litepgo.build_info()` reports what a build contains.

- macOS: Apple Accelerate's LLT and LDLT (`LITEPGO_WITH_ACCELERATE`, on by default).
- x86-64 Linux: Intel oneMKL 2026.1 PARDISO's LLT and pivoted LDLT (`LITEPGO_WITH_MKL`, on by default). Configuring downloads the pinned `mkl` and `mkl-include` wheels from PyPI (about 225 MB) and unpacks them once into `build/_cache/` (`LITEPGO_MKL_CACHE_DIR`), shared by every preset; `-DLITEPGO_MKL_ROOT=/opt/intel/oneapi/mkl/2026.1` uses a local oneMKL 2026.1 install instead. MKL runs on its oneTBB threading layer over litepgo's own TBB runtime, so `tbb::task_arena` and `tbb::global_control` bound PARDISO like every other parallel loop; `MKL_NUM_THREADS` has no effect. Eigen's dense code does not use MKL.

Native IPC contact is provided by `litepgo::contact` and included in the `litepgo::litepgo` aggregate, with no IPC Toolkit dependency.
The optional `litepgo::ipctk` target (`LITEPGO_WITH_IPC_TOOLKIT`, on by default) retains the IPC Toolkit 1.6.1 backend.
It reuses litepgo's Eigen and TBB and fetches its dependencies into `build/_cache/cpm/` (`LITEPGO_CPM_CACHE_DIR`).
`-DLITEPGO_WITH_IPC_TOOLKIT=OFF` skips the toolkit and its dependencies; native C++ and Python contact remain available.



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
