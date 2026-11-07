# Torch material tutorial

Start with a local material formula and finish by designing a plastic state through
native equilibrium. The seven lessons use CPU float64 and run independently
from a fresh kernel. You need basic Python, arrays, and derivatives; no prior knowledge
of litepgo's response or equilibrium interfaces is assumed.

## Learning path

| Lesson | New idea | What you check |
| --- | --- | --- |
| [1. Energy and stress](./01_density.md) | Differentiate a row-independent density with respect to deformation and material inputs | Stress and stress derivatives versus finite differences; relaxed state; loading curves |
| [2. Native FEM assembly](./02_fem.md) | Put that density into a material class | Energy, gradient, and exact Hessian versus native Neo-Hookean |
| [3. Fixed-pose parameter fitting](./03_fixed_pose_fit.md) | Explicit parameters and the outer Torch response bridge | Loss gradient and recovery of physical parameters |
| [4. Equilibrium differentiation](./04_equilibrium_fit.md) | Solve for the pose; implicit rather than unrolled derivatives | Total loss gradient and equilibrium residual |
| [5. Learn a neural density](./05_neural_density.md) | Train an invariant energy network on native Neo-Hookean data and export its weights | Held-out energy, stress, and tangent errors, plus extrapolation |
| [6. Fit the height of a bunny's ear under load](./06_neural_equilibrium_fit.md) | Optimize the lesson 5 network so the pulled bunny reaches a target height | Adjoint gradient, peak height error, and force balance |
| [7. Design a hanging bunny's plastic state](./07_plastic_gravity_design.md) | Fit a local plastic stretch so the bunny hangs at a target height under gravity | Height sensitivity, force balance, and different plastic states with the same lowest point |

Lesson 2 assembles tetrahedral and hexahedral bunny meshes with the same Neo-Hookean law.
Lessons 3 and 4 use the tetrahedral bunny. Lesson 4 clamps the lower part and solves
on the tetrahedra incident to free vertices. Lesson 5 learns an energy density
from local samples and exports its weights. Lesson 6 defines the corresponding
FEM material, pulls on the taller bunny ear, and updates the weights to match
the desired height under the prescribed load. Lesson 7 hangs the bunny from its
ear tips and adjusts a plastic band beneath the clamps to place its lowest point.

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
and experiment. Open the notebooks with the repository root or `tutorials/torch/`
as the working directory so they can locate the meshes in `assets/`.

If using an existing JupyterLab installation, you can explicitly register the project
kernel yourself:

```sh
uv run --no-sync python -m ipykernel install --user --name litepgo --display-name "Python (litepgo)"
```

Then select **Python (litepgo)**. The automated renderer never installs a global
kernelspec: it creates a temporary one pointing at its own Python interpreter.

Each lesson page offers the notebook and its input files. Keep the downloaded
files in the notebook's working directory and use an environment with a built
litepgo. Lesson 5 refreshes the bundled `trained_theta.npy` used by lesson 6.
The website displays output from the executed notebooks.

## Execute and render locally

```sh
# One lesson, from a clean kernel:
uv run --locked --group tutorials python scripts/torch_tutorials.py 01_density

# All seven lessons:
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

`--no-execute` requires a successful cache whose source notebook and declared input
hashes still match. It does not detect changes to the compiled library or installed
dependencies: rerun execution after changing those. Formal builds never reuse the cache.

## Validation and CI

Each code cell has a 180-second timeout by default (`--timeout` changes it). Uncaught
exceptions, numerical assertion failures, wrong environments, and timeouts fail the
build. Failed executed copies are retained for diagnosis. Expected errors, if added as
teaching exercises, must be handled explicitly within their cell.

The documentation workflow installs locked dependencies, builds `litepgo_python`,
tests the execution/export infrastructure, then executes all seven notebooks through
the normal documentation build. It also checks Markdown rendering, seven built HTML
pages, plots through the rendered lessons, and notebook and input downloads. On failure,
executed notebook copies are uploaded as a CI artifact. The other test workflows do
not require the tutorial dependency group.

```sh
uv run --locked --group docs python -m pytest -q tests/python/test_torch_tutorials.py
```

The lessons use fixed examples and numerical tolerances, not exact printed
values or performance thresholds. L-BFGS can evaluate rejected line-search trials;
its plotted closure history need not be monotone. A small observation loss does not
generally establish unique parameter recovery. Neural stability outside the small
teaching setup, GPU execution, double backward through native responses, and shell
fitting are beyond this introductory path.
