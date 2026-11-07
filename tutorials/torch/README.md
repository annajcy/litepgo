# Torch material notebooks

Read the [learning path and setup](../../docs/tutorials/torch/index.md), then open
`01_density.ipynb` through `06_neural_equilibrium_fit.ipynb` in order. Each notebook
runs independently using the repository's `.venv` kernel. Keep `common.py` beside
them and use this directory as the notebook working directory.

From the repository root:

```sh
uv sync --locked --group tutorials
cmake --preset release
cmake --build --preset release --target litepgo_python
uv run --locked --group tutorials python scripts/torch_tutorials.py
```

This executes each notebook in a new kernel and generates its documentation page,
plots and download copy. To run one lesson, append its stem, e.g. `01_density`.
Outputs are cached under `build/torch-tutorials/`; failed executions are saved there
as well. Source notebooks should be committed with outputs cleared and execution
counts reset. Do not maintain a second `.py` or Markdown copy of each lesson.

`common.py` contains geometry only. The constitutive formulas, parameter mapping,
equilibrium factories and optimization steps stay visible in the notebooks.
The docs build always re-executes all lessons. `--no-execute` is a local export
shortcut, and refuses missing, failed or source-stale caches.
