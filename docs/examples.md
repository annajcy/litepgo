# Example catalog

Every example page explains the problem, its mathematical formulation, the API
construction, run commands, expected checks, and limitations. Full maintained
source is embedded in expandable sections on that page. C++ targets match the
filename stem; executable paths mirror `examples/` under `build/release/`.

Start with [custom scalar energies](examples/energy.md#custom-scalar-energies),
then the [Newton chain](examples/solvers.md#constrained-chain). For simulation,
continue to [elastic equilibrium](examples/equilibrium.md), then
[plastic fields](examples/plasticity.md) or [contact](examples/contact.md).
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

All 26 native examples have Python counterparts. Four additional Torch programs
are Python-only; `uv sync --locked` installs their PyTorch dependency along with
the other project dependencies. Asset inputs are bundled; generated-grid examples
need no input files. Output paths and overwrite behavior are documented per case.
A successful numerical run is indicated by exit status zero, not identical printed
last digits on every platform.

## Choose a task

| Area | Program | What it teaches |
| --- | --- | --- |
| Energy | [`custom_energy`](examples/energy.md#custom-scalar-energies) | Fixed/dynamic scalar derivatives and sparse structure |
| Energy | [`energy_composition`](examples/energy.md#energy-composition) | Weighted sums and affine derivative pullback |
| Energy | [`custom_response`](examples/energy.md#custom-vector-responses) | Fixed/dynamic vector residuals and Jacobians |
| Energy | [`mesh_response`](examples/energy.md#batched-material-force-fitting) | Batched fixed-pose material-force calibration |
| Solvers | [`newton_chain`](examples/solvers.md#constrained-chain) | Hard constraints and a complete Newton solve |
| Solvers | [`gauss_newton_fit`](examples/solvers.md#exponential-curve-fitting) | Analytic nonlinear least squares |
| Mesh | [`laplacian_optimization`](examples/mesh.md#laplacian-smoothing-and-editing) | Edge/one-ring assembly, smoothing and editing |
| Mesh | [`arap`](examples/mesh.md#arap-local-global-deformation) | Rotation fitting and constrained global solves |
| Equilibrium | [`tet_p1_corner_pull`](examples/equilibrium.md#tet-p1-corner-pull) | Tetrahedral elastic equilibrium |
| Equilibrium | [`hex_q1_corner_pull`](examples/equilibrium.md#hex-q1-corner-pull) | Trilinear hex equilibrium |
| Equilibrium | [`hex_hermite_corner_pull`](examples/equilibrium.md#hex-hermite-corner-pull) | Physical derivative DOFs and high-order displacement |
| Equilibrium | [`shell_saddle_pull`](examples/equilibrium.md#shell-saddle-pull) | Membrane/bending energy with soft corner loads |
| Plasticity | [`tet_p1_plastic_twist`](examples/plasticity.md#tet-p1-plastic-twist) | Compatible plastic fields and incremental equilibrium |
| Plasticity | [`hex_q1_plastic_twist`](examples/plasticity.md#hex-q1-plastic-twist) | Q1 plastic rest-state change |
| Plasticity | [`hex_hermite_plastic_twist`](examples/plasticity.md#hex-hermite-plastic-twist) | C1 plastic fields and sampled Hermite output |
| Plasticity | [`shell_plastic_field`](examples/plasticity.md#differential-shell-growth) | Incompatible per-face natural geometry |
| Contact | [`tet_p1_box_stack`](examples/contact.md#pressed-tet-box-stack) | Boundary mapping, native IPC, and changing sparsity |
| Spatial | [`spatial_query`](examples/spatial.md#spatial-hash-queries) | Point/region/filtered box-pair queries |
| Spatial | [`point_embedding`](examples/spatial.md#point-embedding) | Location and interpolation for P1/Q1/Hermite |
| Fitting | [`shell_fixed_pose_material_fit`](examples/fitting.md#fixed-pose-shell-material) | Shared shell material from fixed-pose forces |
| Fitting | [`volume_fixed_pose_fp_fit`](examples/fitting.md#fixed-pose-volume-plastic-stretches) | Three diagonal solid plastic stretches |
| Fitting | [`shell_fixed_pose_fp_fit`](examples/fitting.md#fixed-pose-shell-plastic-stretches) | Two in-plane shell plastic stretches |
| Fitting | [`volume_equilibrium_material_fit`](examples/fitting.md#equilibrium-volume-material) | Material from equilibrium reaction experiments |
| Fitting | [`shell_equilibrium_material_fit`](examples/fitting.md#equilibrium-shell-material) | Material-dependent state and reaction sensitivities |
| Fitting | [`volume_equilibrium_fp_fit`](examples/fitting.md#equilibrium-volume-plastic-stretches) | Plastic stretches from shapes and reactions |
| Fitting | [`shell_equilibrium_fp_fit`](examples/fitting.md#equilibrium-shell-plastic-stretches) | Shell natural geometry through equilibrium |
| Torch | [`volume_equilibrium_material_fit_torch`](examples/torch.md#volume-material-with-l-bfgs) | Volume material recovery with native autograd |
| Torch | [`shell_equilibrium_material_fit_torch`](examples/torch.md#shell-material-with-l-bfgs) | Shell material recovery with native autograd |
| Torch | [`volume_equilibrium_fp_fit_torch`](examples/torch.md#volume-plastic-stretches-with-l-bfgs) | Three-stretch inverse equilibrium with L-BFGS |
| Torch | [`shell_equilibrium_fp_fit_torch`](examples/torch.md#shell-plastic-stretches-with-l-bfgs) | Two-stretch shell inverse equilibrium with L-BFGS |

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
