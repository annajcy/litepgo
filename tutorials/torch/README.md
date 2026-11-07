# Torch material notebooks

Read the [learning path and setup](../../docs/tutorials/torch/index.md), then open
`01_density.ipynb` through `07_plastic_gravity_design.ipynb` in order. Each notebook
runs independently using the repository's `.venv` kernel. Run from this directory
or the repository root to load the bunny meshes from `assets/`. If you download a
notebook from the docs, put its linked files in the notebook's working directory.
Lesson 5 refreshes the bundled `trained_theta.npy`. Lesson 6 defines the FEM
material and loads those weights for the equilibrium fit.

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

The constitutive formulas, geometry setup, parameter mapping, equilibrium factories,
and optimization steps stay visible in the notebooks. The docs build always
re-executes all lessons. `--no-execute` rejects missing or failed caches and detects
changes to the notebook or its declared input files.
