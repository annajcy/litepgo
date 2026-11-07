# Torch material tutorial

Start with a local material formula and finish by fitting network weights through
native equilibrium. The six lessons use small CPU float64 problems and run independently
from a fresh kernel. You need basic Python, arrays, and derivatives; no prior knowledge
of litepgo's response or equilibrium interfaces is assumed.

## Learning path

| Lesson | New idea | What you check |
| --- | --- | --- |
| [1. Energy and stress](./01_density.md) | Write a row-independent Torch density | Stress versus finite differences; rest state; loading curves |
| [2. Native FEM assembly](./02_fem.md) | Put that density into a material class | Energy, gradient, and exact Hessian versus native Neo-Hookean |
| [3. Fixed-pose parameter fitting](./03_fixed_pose_fit.md) | Explicit parameters and the outer Torch response bridge | Loss gradient and recovery of physical parameters |
| [4. Equilibrium differentiation](./04_equilibrium_fit.md) | Solve for the pose; implicit rather than unrolled derivatives | Total loss gradient and equilibrium residual |
| [5. A small neural density](./05_neural_density.md) | A smooth 2–2–1 network adds a reference-corrected density | Rest state, rotation invariance, tangent, and native assembly |
| [6. Fit network weights](./06_neural_equilibrium_fit.md) | Network weights enter the explicit parameter slot | Mixed derivatives, independent rows, and equilibrium fitting |

Lessons 2–4 keep the same two-tetrahedron mesh and Neo-Hookean law. Lessons 5–6 use
an analytic baseline plus a tiny network. The hidden layer is fixed; only two output
weights are fitted in lesson 6. This avoids pretrained checkpoints and full-network
parameter management while keeping every important step visible.

The **notebook is the course source**. The site executes and exports
`tutorials/torch/*.ipynb`; the Markdown pages and figures here are generated.
The existing [Torch examples](../../examples/torch/index.md) remain advanced/reference
programs, and [benchmarks](../../benchmarks/torch-material.md) cover larger workloads.

## Set up your environment

From the repository root:

```sh
uv sync --locked --group tutorials
cmake --preset release
cmake --build --preset release --target litepgo_python
uv run --no-sync python -c "import litepgo; print(litepgo.build_info())"
```

See [build setup](../../guide/build.md) for native prerequisites and alternative
builds. The CMake Python target registers this checkout in the project environment.
Tutorial dependencies are optional; they are not dependencies of the library itself.
The `docs` dependency group includes them so documentation CI uses the same lockfile.

Open a notebook in VS Code with its Jupyter extension and choose this repository's
`.venv` Python as the kernel. Use **Restart Kernel and Run All** first. Then edit cells
and experiment. The notebook working directory should be `tutorials/torch/` so
`from common import make_geometry` finds the adjacent helper. If your editor starts
in the repository root, change the notebook working directory to that folder.

If using an existing JupyterLab installation, you can explicitly register the project
kernel yourself:

```sh
uv run --no-sync python -m ipykernel install --user --name litepgo --display-name "Python (litepgo)"
```

Then select **Python (litepgo)**. The automated renderer never installs a global
kernelspec: it creates a temporary one pointing at its own Python interpreter.

Every course page offers notebook and helper downloads. Keep `common.py` beside the
downloaded notebooks and use an environment with a built litepgo. The helper contains
only geometry setup; materials, parameter maps, factories, loss, and optimization
are visible in the notebooks. The website shows static executed output, not a live
Python kernel.

## Execute and render locally

```sh
# One lesson, from a clean kernel:
uv run --locked --group tutorials python scripts/torch_tutorials.py 01_density

# All six lessons:
uv run --locked --group tutorials python scripts/torch_tutorials.py

# Re-export a matching execution cache without running cells again:
uv run --locked --group tutorials python scripts/torch_tutorials.py --no-execute

# Complete documentation build (executes all lessons again):
uv run scripts/docs.py build
```

To select a different registered build explicitly, prefix the renderer command with
`PYTHONPATH=build/release/python` on POSIX. Relative entries are resolved before the
kernel changes its working directory. The renderer checks that the kernel uses its
Python environment and imports litepgo from this checkout. No automatic build-path
fallback or hidden system Python kernel is used.

The source notebooks are committed with no outputs. Executed copies go to
`build/torch-tutorials/`; generated pages and plot assets go to this directory, and
source downloads go to `docs/public/tutorial-notebooks/torch/`. All generated files
are ignored by Git. Edit the notebook, not its generated `.md` page.

`--no-execute` requires a successful cache whose source notebook and `common.py`
hashes still match. It does not detect changes to the compiled library or installed
dependencies: rerun execution after changing those. Formal builds never reuse the cache.

## Validation and CI

Each code cell has a 180-second timeout by default (`--timeout` changes it). Uncaught
exceptions, numerical assertion failures, wrong environments, and timeouts fail the
build. Failed executed copies are retained for diagnosis. Expected errors, if added as
teaching exercises, must be handled explicitly within their cell.

The documentation workflow installs locked dependencies, builds `litepgo_python`,
tests the execution/export infrastructure, then executes all six notebooks through
the normal documentation build. It also checks Markdown rendering, six built HTML
pages, plots through the rendered lessons, and notebook/helper downloads. On failure,
executed notebook copies are uploaded as a CI artifact. The other test workflows do
not require the tutorial dependency group.

```sh
uv run --locked --group docs python -m pytest -q tests/python/test_torch_tutorials.py
```

The lessons use fixed small problems and numerical tolerances, not exact printed
values or performance thresholds. L-BFGS can evaluate rejected line-search trials;
its plotted closure history need not be monotone. A small observation loss does not
generally establish unique parameter recovery. Neural stability outside the small
teaching setup, GPU execution, double backward, shell/plastic fitting, and full-network
training are beyond this introductory path.
