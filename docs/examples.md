# Example catalog

Every example page explains the problem, its mathematical formulation, the API
construction, run commands, expected checks, and limitations. Full maintained
source is embedded in expandable sections on that page. C++ targets match the
filename stem; executable paths mirror `examples/` under `build/release/`.

Start with [custom scalar energies](examples/energy/custom-energy.md),
then the [Newton chain](examples/solvers/newton-chain.md). For simulation,
continue to [elastic equilibrium](examples/equilibrium/index.md), then
[plastic fields](examples/plasticity/index.md) or [contact](examples/contact/index.md).
Read [inverse problems](guide/solver/inverse.md) before choosing a fitting experiment.

## Prepare and run

From the repository root, create the environment, configure release, and build
an example and Python extension. For example:

```sh
uv sync --locked
cmake --preset release
cmake --build build/release --target custom_energy litepgo_python
build/release/examples/energy/scalar/custom_energy
PYTHONPATH=build/release/python uv run --no-sync python examples/energy/scalar/custom_energy.py
```

All cataloged native examples have Python counterparts. Seven additional Torch programs
are Python-only; `uv sync --locked` installs their PyTorch dependency along with
the other project dependencies. Asset inputs are bundled; generated-grid examples
need no input files. Output paths and overwrite behavior are documented per case.
A successful numerical run is indicated by exit status zero, not identical printed
last digits on every platform.

## Choose a task

| Area | Program | What it teaches |
| --- | --- | --- |
| FEM | [`sampling_response`](examples/fem/sampling-response.md) | Coefficient maps to fixed-point values and reference derivatives |
| FEM | [`interpolated_response`](examples/fem/interpolated-response.md) | Fixed coefficients and exact spatial derivatives |
| FEM | [`tet_p1_basis`](examples/fem/tet-p1-basis.md) | Shape functions, target sampling at vertices, and scalar Field reconstruction |
| FEM | [`hermite_interpolation`](examples/fem/hermite-interpolation.md) | Value/derivative samples and Hermite Field reconstruction in 2D/3D |
| FEM | [`quadrature_integration`](examples/fem/quadrature-integration.md) | Reference integration in 1D/2D/3D and simplex area/volume weights |
| FEM | [`numpy_neohookean`](examples/fem/numpy-neohookean.md) | A volumetric material written in NumPy, assembled by native Hermite FEM |
| FEM | [`shell_uniform_fp`](examples/fem/shell-uniform-fp.md) | Uniform Fp=diag(1,0.5,1), free shell contraction and affine target check |
| FEM | [`shell_fp_free_relaxation`](examples/fem/shell-fp-free-relaxation.md) | Free Koiter–StVK shell relaxation under fixed nonuniform Fp |
| FEM | [`numpy_koiter_shell`](examples/fem/numpy-koiter-shell.md) | A shell material written in NumPy, with force responses and equilibrium |
| Energy | [`custom_energy`](examples/energy/custom-energy.md) | Fixed/dynamic scalar derivatives and sparse structure |
| Energy | [`mesh_energy_subclass`](examples/energy/mesh-energy-subclass.md) | Subclass assembly and spring equilibrium |
| Energy | [`energy_composition`](examples/energy/energy-composition.md) | Weighted sums and affine derivative pullback |
| Energy | [`mesh_response_subclass`](examples/energy/mesh-response-subclass.md) | Subclass assembly, JVP/VJP, and parameter fitting |
| Energy | [`custom_response`](examples/energy/custom-response.md) | Fixed/dynamic vector residuals and Jacobians |
| Energy | [`mesh_response`](examples/energy/mesh-response.md) | Batched fixed-pose material-force calibration |
| Solvers | [`newton_chain`](examples/solvers/newton-chain.md) | Hard constraints and a complete Newton solve |
| Solvers | [`thread_limit`](examples/solvers/thread-limit.md) | Scoped TBB limits, nesting, and exception cleanup |
| Solvers | [`gauss_newton_fit`](examples/solvers/gauss-newton-fit.md) | Analytic nonlinear least squares |
| Mesh | [`laplacian_optimization`](examples/mesh/laplacian-optimization.md) | Edge/one-ring assembly, smoothing and editing |
| Mesh | [`arap`](examples/mesh/arap.md) | Rotation fitting and constrained global solves |
| Mesh Processing | [`constrained_remesh`](examples/mesh-processing/constrained-remesh.md) | CGAL remeshing with fixed positions, protected boundaries, and a self-intersection check |
| Equilibrium | [`tet_p1_corner_pull`](examples/equilibrium/tet-p1-corner-pull.md) | Tetrahedral elastic equilibrium |
| Equilibrium | [`hex_q1_corner_pull`](examples/equilibrium/hex-q1-corner-pull.md) | Trilinear hex equilibrium |
| Equilibrium | [`hex_hermite_corner_pull`](examples/equilibrium/hex-hermite-corner-pull.md) | Physical derivative DOFs and high-order displacement |
| Equilibrium | [`shell_saddle_pull`](examples/equilibrium/shell-saddle-pull.md) | Membrane/bending energy with soft corner loads |
| Plasticity | [`tet_p1_plastic_twist`](examples/plasticity/tet-p1-plastic-twist.md) | Compatible plastic fields and incremental equilibrium |
| Plasticity | [`hex_q1_plastic_twist`](examples/plasticity/hex-q1-plastic-twist.md) | Q1 plastic rest-state change |
| Plasticity | [`hex_hermite_plastic_twist`](examples/plasticity/hex-hermite-plastic-twist.md) | C1 plastic fields and sampled Hermite output |
| Plasticity | [`shell_plastic_field`](examples/plasticity/shell-plastic-field.md) | Incompatible per-face natural geometry |
| Contact | [`tet_p1_box_stack`](examples/contact/tet-p1-box-stack.md) | Boundary mapping, native IPC, and changing sparsity |
| Contact | [`tet_p1_plate_compression`](examples/contact/tet-p1-plate-compression.md) | Elasticity, IPC self-contact, and two one-sided plate penalties |
| Contact | [`contact_area_weighting`](examples/contact/area-weighting.md) | Contact energy and force under mesh refinement, with area weighting off/on |
| Contact | [`contact_snapshot`](examples/contact/contact-snapshot.md) | Export every active contact record and plot individual stencils and distances |
| Contact | [`contact_pair_weights`](examples/contact/pair-weights.md) | Three primitive groups scale contact energy and force by 0.5, 2 and 0 |
| Spatial | [`spatial_query`](examples/spatial/spatial-query.md) | Point/region/filtered box-pair queries |
| Spatial | [`point_embedding`](examples/spatial/point-embedding.md) | Location and interpolation for P1/Q1/Hermite |
| Fitting | [`shell_fixed_pose_material_fit`](examples/fitting/shell-fixed-pose-material-fit.md) | Shared shell material from fixed-pose forces |
| Fitting | [`volume_fixed_pose_fp_fit`](examples/fitting/volume-fixed-pose-fp-fit.md) | Three diagonal solid plastic stretches |
| Fitting | [`shell_fixed_pose_fp_fit`](examples/fitting/shell-fixed-pose-fp-fit.md) | Two in-plane shell plastic stretches |
| Fitting | [`volume_equilibrium_material_fit`](examples/fitting/volume-equilibrium-material-fit.md) | Material from equilibrium reaction experiments |
| Fitting | [`shell_equilibrium_material_fit`](examples/fitting/shell-equilibrium-material-fit.md) | Material-dependent state and reaction sensitivities |
| Fitting | [`volume_equilibrium_fp_fit`](examples/fitting/volume-equilibrium-fp-fit.md) | Plastic stretches from shapes and reactions |
| Fitting | [`shell_equilibrium_fp_fit`](examples/fitting/shell-equilibrium-fp-fit.md) | Shell natural geometry through equilibrium |
| Torch | [`neural_neohookean`](examples/torch/neural-neohookean.md) | Learn an invariant energy and validate stress, tangent, and equilibrium |
| Torch | [`neural_equilibrium_fit`](examples/torch/neural-equilibrium-fit.md) | Fit two network weights through equilibrium with checked implicit gradients |
| Torch | [`torch_neohookean`](examples/torch/torch-neohookean.md) | Torch density, native FEM assembly, and derivative comparison |
| Torch | [`torch_equilibrium_response`](examples/torch/torch-equilibrium-response.md) | Physical theta/Fp Jacobians and equilibrium JVP/VJP |
| Torch | [`torch_shell_material_fit`](examples/torch/torch-shell-material-fit.md) | Torch shell density fitted from equilibrium reactions |
| Torch | [`volume_equilibrium_material_fit_torch`](examples/torch/volume-equilibrium-material-fit-torch.md) | Volume material recovery with native autograd |
| Torch | [`shell_equilibrium_material_fit_torch`](examples/torch/shell-equilibrium-material-fit-torch.md) | Shell material recovery with native autograd |
| Torch | [`volume_equilibrium_fp_fit_torch`](examples/torch/volume-equilibrium-fp-fit-torch.md) | Three-stretch inverse equilibrium with L-BFGS |
| Torch | [`shell_equilibrium_fp_fit_torch`](examples/torch/shell-equilibrium-fp-fit-torch.md) | Two-stretch shell inverse equilibrium with L-BFGS |

## Read fitting names correctly

`fixed_pose` holds displacement fixed and fits force predictions. `equilibrium`
re-solves free displacement at every parameter trial and includes its sensitivity.
`material` fits constitutive parameters; `fp` fits a restricted plastic-state
parameterization. Volume and shell names select different constitutive/natural
geometry models. `torch` selects the outer optimizer/autograd interface and does
not change the inner equilibrium formulation.

Examples with recovered synthetic parameters establish numerical correctness and
local identifiability for those experiments. They do not establish uniqueness for
arbitrary experimental data or supply a general plastic-evolution model.

## Reading the source

Small API demonstrations stay linear: define the model, call it, and inspect the
result. Larger fitting examples separate problem construction, physical factories,
measurement, normalized residual, and the optimizer. Read `main` first for the
experiment, then follow those named functions for the formulas. Fixed-pose examples
need no equilibrium stage.

Torch fitting drivers import the adjacent native Python example's physical model;
they show parameter tensors, L-BFGS, and the response bridge. Keep both files
together. Derivative, rank, and recovery checks belong to `tests/python`, while
input validation and solver failure handling remain in the runnable examples.
Simulation and plotting examples retain their setup → solve → export/plot flow;
the batch comparison example also verifies equivalence before timing.
