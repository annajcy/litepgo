# Learning a Neo-Hookean material

[Back to PyTorch examples](./index.md)

This Python example learns an energy from native Neo-Hookean energy and first
Piola stress samples, then freezes the network and uses it as the Material of
`fem.VolumetricFemEnergy` through a `TorchVolumetricMaterial` subclass
(`fem_material` in the source). No analytic Neo-Hookean energy is added to the network.
The physical parameters are fixed at lambda = 2 and mu = 3; other `theta` values
are rejected. Network weights are trained directly with Torch, independently of
the native equilibrium response parameter interface.
For training explicit network weights from equilibrium observations, see
[the two-weight equilibrium fit](./neural-equilibrium-fit.md). That separate
example adds a fixed analytic baseline and needs no pretrained checkpoint.

This is an end-to-end learning and FEM integration example: learn a density,
check its derivatives on held-out data, then solve equilibrium with frozen weights.
Start with the default energy/stress-only run. The directional tangent experiment
is an optional comparison, not a required improvement or a production material model.

Read `InvariantMaterial` for the density, `train` for the loss and optimizer,
`errors` for constitutive comparisons, and `equilibrium_errors` for native/Torch
FEM integration. The model lives in the example; it is not a new public library API.

Legacy checkpoints with an output bias remain loadable: that bias cancels in the
reference correction and is discarded. Reported experiments used the original
parameterization; new training runs need not reproduce their last digits.

## Model

For positive elastic determinant, define

$$
F_e=FF_p^{-1},\quad C=F_e^TF_e,\quad J=\det F_e,
$$

$$
z=(J^{-2/3}\operatorname{tr}C-3,
J^{-4/3}((\operatorname{tr}C)^2-\operatorname{tr}(C^2))/2-3,\log J).
$$

A float64 MLP with widths 3–32–32–1 and Softplus activations (1,216 parameters)
produces a scalar $f_\omega(z)$. The final density is

$$
W(F,F_p)=\det(F_p)\left[f_\omega(z)-f_\omega(0)
-\partial_{z_3}f_\omega(0)\log J\right].
$$

Invariants enforce objectivity and isotropy. The correction enforces zero energy
and stress at $F_e=I$; its origin slope stays differentiable with respect to network
weights during training. The density includes the plastic volume factor because
the FEM adapter integrates over the original reference volume.

## Training and independent checks

The dataset contains 10,000 training and 2,000 validation deformations, with
principal stretches sampled uniformly in [0.7, 1.3] and independent rotations.
Native `NeoHookean` supplies energy and stress labels. The loss sums energy and
stress mean-square errors, each normalized by its training-label RMS. Training
uses 1,500 Adam minibatches followed by a budget of 300 full-batch L-BFGS iterations;
line search may evaluate the objective multiple times. Validation selects the
best checkpoint. Model/minibatch seed is 42; dataset seeds are 11 and 12.

Evaluation uses 512 independent random deformations (seed 13), four 61-point
loading paths, and 256 samples with a nonidentity plastic matrix (seed 14).
The paths are constrained uniaxial and biaxial stretches, volumetric stretch,
and simple shear. In the default baseline, tangent stiffness is **not** a training label. Native tangent
indices are converted to Torch's row-major order before comparison.

One CPU run of the default command produced these aggregate relative L2 errors:

| Evaluation | Energy | Stress | Tangent |
| --- | ---: | ---: | ---: |
| Random test | 0.023% | 0.070% | 0.276% |
| Nonidentity plastic matrix | 0.022% | 0.072% | 0.277% |
| Volumetric path | 0.046% | 0.158% | 0.510% |

Six constrained two-tetrahedron equilibrium solves (three imposed displacements,
identity and nonidentity plastic matrices) all converged. Maximum relative free
displacement error was 0.0225%, and maximum reaction error was 0.1194%.
Elastic stretches at these solutions stayed inside the training interval.
The complete training and evaluation run took about 18 seconds on the development
machine with one Torch thread; this is an observation, not a portable benchmark.

The script exits successfully only when every material evaluation has energy
and stress relative L2 error below 1% and tangent error below 2%, and all six
solves converge with displacement/reaction errors below 1% and elastic stretches
inside the sampled interval. These are pilot acceptance thresholds, not bounds
on every individual component. The JSON also records maximum absolute errors.

## Run and inspect

Run from the repository root with the configured Python build:

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/neural_neohookean.py
uv sync --locked --group plot
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/plot_neural_neohookean.py
uv run --no-sync python -m pytest -q tests/python/test_neural_material_example.py
```

The training command writes `material.pt` and `metrics.json` under
`output/neural_neohookean`; the plotting command writes `comparison.png` there.
Repeated runs overwrite these outputs. Use `--output PATH` for a separate training
run, and pass `PATH/material.pt` to the plotting command. Outputs are ignored by Git.
The plot compares energy, one stress component, and its corresponding tangent
component along each loading path.

To reuse the checkpoint from a script alongside the example:

```python
import torch
from neural_neohookean import InvariantMaterial, fem_material

material = InvariantMaterial()
checkpoint = torch.load("output/neural_neohookean/material.pt", weights_only=True)
material.load_state_dict(checkpoint["state_dict"])
material.eval().requires_grad_(False)
# fem_material(material) is a TorchVolumetricMaterial class for the Material slot,
# e.g. fem.VolumetricFemEnergy[fem_material(material), fem.TetP1Basis,
# fem.TetCentroidQuadrature], with theta fields (2, 3).
```

## Directional tangent supervision comparison

The optional `--tangent-supervision` adds 128 independent deformation samples
(seed 21), each with one unit random matrix direction $v$. It minimizes an
additional mean-square error in $Av$, normalized by the training target RMS,
with coefficient 1. A separate set of 128 samples (seed 22) adds the same term
to validation checkpoint selection. The energy/stress datasets, initialization,
optimizer budgets, and independent evaluation sets remain the same as baseline.
Validation losses from the two variants have different definitions and should
not be compared directly.

Torch computes the product as $\nabla_F[(\nabla_FW):v]$, using Hessian symmetry;
it does not assemble a learned 9-by-9 tangent during training. Native labels
are generated once from the native full tangent. Backpropagation through the
product retains the mixed derivatives with respect to network weights. Test
loading paths are not used for supervision or checkpoint selection.

Run the comparison without overwriting the baseline checkpoint:

```sh
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/neural_neohookean.py --tangent-supervision --output output/neural_neohookean_tangent
PYTHONPATH=build/release/python uv run --no-sync python examples/fem/fitting/plot_neural_neohookean.py output/neural_neohookean_tangent/material.pt --baseline output/neural_neohookean/material.pt
```

Observed results for the same initialization:

| Metric | Energy/stress only | With directional tangent labels |
| --- | ---: | ---: |
| Random energy relative L2 | 0.0226% | 0.0129% |
| Random stress relative L2 | 0.0698% | 0.0332% |
| Random full tangent relative L2 | 0.2764% | 0.0865% |
| Shear path full tangent relative L2 | 0.0929% | 0.0443% |
| Shear $A_{1212}$ maximum relative error | 0.0981% | 0.1377% |
| Maximum equilibrium reaction relative error | 0.1194% | 0.0319% |
| Training and evaluation wall time | 18.1 s | 20.8 s |

Both runs pass the same acceptance criteria, including all six equilibrium solves.
The aggregate tangent improves, but the specific shear component has larger errors
at the path endpoints. Directional supervision is not a pointwise guarantee; the
comparison plot preserves a shared axis for both networks so this remains visible.
For identity plastic deformation and simple shear $F=I+\gamma e_1e_2^T$,
the native material has $W=\mu\gamma^2/2$, $P_{12}=\mu\gamma$, and
$A_{1212}=\mu=3$. A smooth MLP approximates this quadratic without enforcing it
exactly. Energy and stress can be accurate while their next derivative still
varies slightly. The plot magnifies this small variation with a narrow tangent
axis; it is not a large stress error.

Random directional labels constrain combinations of tangent entries in an
average loss. They do not guarantee a smaller maximum error for one entry along
one held-out path. This run demonstrates that distinction; it does not establish
whether more samples or more optimization would remove the remaining error.
Timings are individual development-machine runs, not repeated benchmarks.

Tests check the reference state, rotation invariance, stress-loss weight gradients,
plastic volume factor, checkpoint round trips, native tangent packing, directional
products and their weight gradients, and small training runs with both losses.
The full default training experiment is deliberately outside pytest.

This ordinary MLP does not guarantee polyconvexity, positive stiffness, or reliable
extrapolation. Results cover one fixed material and one initialization, not a
family of material parameters. Training network weights through
`StaticEquilibriumResponse` still needs an explicit optimization-vector-to-weight
mapping; this example trains on constitutive samples and evaluates frozen-weight
FEM solves.

::: details Complete training and evaluation source

<<< ../../../examples/fem/fitting/neural_neohookean.py

:::

::: details Plotting source

<<< ../../../examples/fem/fitting/plot_neural_neohookean.py

:::
