# Torch material performance

[All benchmarks](./index.md)

`benchmarks/torch_material.py` separates constitutive evaluation, native FEM
assembly, and implicit equilibrium gradients. It compares native Neo-Hookean,
the identical Torch formula, and a fixed Neo-Hookean baseline plus the existing
3–32–32–1 invariant MLP. The archived baseline used the pointwise functional-call approach
from [network-weight fitting](../examples/torch/neural-equilibrium-fit.md), extended
to 2, 8, or 32 active weights in the final layer. At their reference values, all
three active-weight counts describe the **same** neural material.

The current driver reuses the stateless batched material from the fitting example.
The recorded baseline below is preserved as historical evidence; its lock and
pointwise-call discussion describes that older implementation, not the current hot path.

## Run and outputs

Use the release build on macOS or Linux:

```sh
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/torch_material.py
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/torch_material.py --profile full --timeout 90 --output output/torch_material_full
PYTHONPATH=build/release/python uv run --no-sync python -m pytest -q tests/python/test_torch_material_benchmark.py tests/python/test_neural_equilibrium_fit_example.py
```

The default pilot has smaller meshes and a 120-second per-case timeout. Full uses
the same five measured repetitions unless `--repeats` is supplied. The commands
above produce:

- `environment.json`: command, UTC start, hardware/OS, Python, Git state, source
  hashes and thread environment. Each result also records Torch/NumPy versions and
  native `build_info()`.
- `results.jsonl`: all measured samples, first-call time, setup time, peak resident
  memory, diagnostic Newton status/iterations, and errors or timeouts.
- `timings.csv`: compact summaries of those records, including failed cases.

Each case runs in a fresh child process. It is stopped at the timeout even if it is
inside a slow derivative calculation. Completed operation records survive a later
case timeout; missing operations have **no timing estimate**. Exit status is 0 for
complete success, 1 if any case errors, or 2 for timeouts without other errors.
Output files in the chosen directory are replaced on rerun.

## Workloads

| Layer | Operations | Full sweep |
| --- | --- | --- |
| Material | `W`, stress, material Hessian, parameter mixed derivative | 32 / 256 / 2,048 points |
| FEM | Value, gradient, exact sparse Hessian, fixed-state parameter VJP | Connected unit-cube meshes with 162 / 1,296 / 10,368 TetP1 elements |
| Equilibrium | Forward displacement; backward of a displacement loss | 6 / 48 TetP1 elements |

The structured grid uses 6 tetrahedra per cube, so subdivision counts 3/6/12 give
the FEM element counts above. It reuses the validated mesh generator from
`benchmarks/contact_scaling.py`. Geometry, deformation, plastic identity, and
material parameters are deterministic. The prescribed FEM deformation includes
stretch and shear and is identical across all three materials.

The default batch is 256 **elements**, with one material point per element under
centroid quadrature. At 1,296 elements, the full profile also measures batches
1/32/1,024 and a representative 4-thread case. Weight sweeps use 256 material
points, 162-element FEM, and 6-element equilibrium. These are one-factor sweeps,
not the Cartesian product of all combinations. The pilot uses 32 material points,
6/48-element FEM, 6-element equilibrium, and smaller weight-sweep meshes.

At the material layer, the current driver calls
`PointwiseVolumetricAdapter[NeoHookean]().eval(...)` once per batch, with the
pointwise loop running in C++. Timings include the Python binding and array
conversion costs. The Python pointwise loop is used only as an untimed correctness
reference for Torch outputs. Each material call requests one output; Torch second
derivatives still require the first derivative internally.

The archived runs below predate this interface migration. Their native material
rows include a Python pointwise loop and should not be treated as measurements
of the current batch adapter.

For equilibrium, the x=0 face is fixed and the x=1 face receives a fixed total
force `[0.015, -0.01, 0.02]`, distributed equally among its vertices. Each solve
starts from zero displacement, with gradient tolerance `1e-10` and at most 50
iterations. The loss is half the mean square of free displacements. It is nonzero
and exercises weight sensitivity without requiring a separate target solve.
Native/Torch parameters are physical `[lambda, mu]`; neural parameters are network
weights. Their parameter VJPs therefore have different physical meanings.

## Measurement boundaries

All arrays are CPU float64. The first measured call is retained separately.
After its correctness check, two additional warmups precede five measured calls;
the report includes median, min/max, and every sample. These are wall-clock
observations on a shared desktop, without background-process termination.
Small timings include Python and timer overhead.

`setup_ms` includes network/FEM construction and mesh generation/loading, excluding
imports and the separately reported diagnostic solve. Steady FEM calls reuse
constructed objects. The equilibrium factories construct new energies/responses
as part of each solve, so that cost remains inside the forward/backward timings.
The first-call measurement is not a cold-start process measurement: imports and
construction have already occurred, and equilibrium has a prior diagnostic solve.

Backward is timed after constructing a fresh Torch loss graph outside the timed
region. It **includes a fresh equilibrium solve**, exact stiffness linearization,
and the adjoint/parameter VJP. Its separate forward time must be added when
estimating a complete loss-gradient iteration. No Newton iterates are unrolled.
The Newton iteration count comes from a separate solve with identical inputs and
options; internal bridge solves do not expose their traces.

Torch intra-op and TBB limits are both 1 by default and both 4 in the representative
thread comparison. Torch inter-op is 1; OpenMP/MKL/OpenBLAS/Accelerate environment
limits are set to 1 before child imports. Actual backend behavior still depends on
the installed build. This is a combined thread-setting comparison, not a separate
sweep of Torch and TBB thread pools.

Peak RSS comes from `resource.getrusage` and includes native memory. Both the
post-import high-water mark and cumulative process high-water mark are reported.
**The latter includes correctness checks and earlier operations in the same case**;
it is not an isolated per-operation allocation measurement, and subtracting the
two high-water marks does not yield exact working memory. No Python-only allocator
measurement is substituted for native RSS.

## Correctness gates and concurrency

Before steady timing, Torch Neo-Hookean values/derivatives are compared with native
results, and FEM checks compare alternate batch partitions. Neural pointwise
results are checked against independently evaluated rows. Equilibrium checks
convergence and compares the final loss directional derivative with centered
finite differences that re-solve equilibrium. Smoke tests exercise all layers,
raw sample recording, timeout retention, and multi-threaded neural evaluation.
The fitting example tests independently check weight replacement and mixed derivatives.

An exploratory pilot exposed a real shared-model race: `functional_call`
temporarily replaces module parameters, and Torch operations may release the GIL.
Different native batches could therefore overlap those replacements and return
incorrect parameter VJPs. The historical implementation added a per-material lock
around functional model evaluation. It protected only the forward material call;
autograd operated on the resulting graph afterward. This serialization was part of
the historical measurements. The current batched implementation below eliminates
module replacement and the lock entirely. Invalid exploratory
results are excluded from the recorded baseline.

## Native batch-path optimization

On 2026-11-10, the migrated code was committed before optimization as `915bb9d`
(the implementation is in `0bba835`). macOS `sample` captured the native
1,296-element, batch-size-1 workload. The value call stack showed nested TBB
execution inside the outer MeshEnergy batch dispatch, plus allocation/free work.
Release symbols are partially stripped, so the samples do not establish exact
per-source-line costs or a percentage attributable to virtual dispatch.

Two changes preserve the runtime batch-material interface:

- For a one-element batch, run the gather and integration loops directly instead
  of creating nested TBB tasks. Multi-element gathering retains its isolated task
  context, preserving initialization when a sibling batch throws.
- Write deformation gradients, fixed Fp and theta directly into batch buffers.
  This removes the intermediate owning material-input object and its theta copy,
  plus the temporary dynamic deformation vector. Finite-value checks remain.

The optimized full scan completed **42 cases / 152 validated measurements**.
All **505 CTest entries passed**. Six native cases were then run in
legacy/unoptimized/optimized/optimized/unoptimized/legacy order, ten samples per
operation per process. Legacy is `2b524da`; the unoptimized binary was saved before
rebuilding. All versions used the same Python dependencies and benchmark settings.
No compilation or test suite ran during benchmark timing.

[Full samples](/benchmark-data/torch-material-optimized/results.jsonl.txt),
[environment](/benchmark-data/torch-material-optimized/environment.json),
[three-version comparison](/benchmark-data/torch-material-optimized/comparison.csv),
[alternating raw samples](/benchmark-data/torch-material-optimized/rechecks.jsonl.txt),
[before profile](/benchmark-data/torch-material-optimized/before-sample.txt),
[after profile](/benchmark-data/torch-material-optimized/after-sample.txt).
Profiles are qualitative call-stack evidence, not normalized before/after timing;
the faster process can reach a different operation within the sampling window.

Selected single-thread medians (milliseconds, 20 samples per version):

| Workload | Batch | Legacy | Unified, unoptimized | Optimized |
| --- | ---: | ---: | ---: | ---: |
| FEM value, 1,296 elements | 1 | 0.499 | 0.812 | 0.373 |
| FEM gradient, 1,296 elements | 1 | 0.678 | 0.995 | 0.579 |
| FEM Hessian, 1,296 elements | 1 | 1.851 | 2.334 | 1.879 |
| FEM parameter VJP, 1,296 elements | 1 | 0.564 | 0.739 | 0.750 |
| FEM value, 1,296 elements | 256 | 0.298 | 0.319 | 0.257 |
| FEM gradient, 1,296 elements | 256 | 0.463 | 0.496 | 0.446 |
| FEM Hessian, 10,368 elements | 256 | 13.827 | 14.557 | 13.807 |
| Equilibrium backward, 48 elements | 256 | 2.590 | 2.744 | 2.593 |

The one-element-batch energy regression is largely removed: value and gradient
are faster than legacy in these samples; Hessian is about 1.5% above legacy.
The force-response parameter VJP path was not changed and still costs about 33%
more than legacy at batch 1, and about 6–8% more at default batch in these rechecks.
This optimization does not eliminate every migration cost.

The initial four-thread recheck was noisy, including a large apparent VJP
regression in unchanged code. A separate alternating run with 50 samples per
process (100 per version) gave legacy/unoptimized/optimized medians of
0.160/0.241/0.135 ms for value, 0.235/0.377/0.205 ms for gradient,
0.672/0.835/0.753 ms for Hessian and 0.283/0.226/0.264 ms for VJP.
[All four-thread follow-up samples](/benchmark-data/torch-material-optimized/threads-recheck.jsonl.txt)
are retained. Four-thread Hessian therefore still exceeds legacy by about 12%;
these observations are not confidence intervals. Shell performance was not measured.

## Full migration comparison

Recorded on 2026-11-09 on the same Apple M1 Pro, using baseline commit
`2b524da` and the uncommitted runtime-material migration. Both versions were built
with the same compiler, Release `-O3 -DNDEBUG`, Eigen settings, Accelerate backend,
dependency sources and Python environment. The baseline was extracted into an
isolated build; imports were verified using explicit `PYTHONPATH` values.

Each version completed **42 cases and 152 validated measurements**, with five
samples per operation, the same inputs, thread limits and 120-second case timeout.
Neither version had a numerical error or timeout. Full scans ran serially, baseline
first. Seven selected cases were then checked in before/after/after/before order,
with ten samples per operation in each process. No compilation ran during timing.

[Full comparison CSV](/benchmark-data/torch-material-runtime-full/comparison.csv),
[alternating rechecks](/benchmark-data/torch-material-runtime-full/rechecks.csv),
[recheck raw samples](/benchmark-data/torch-material-runtime-full/rechecks.jsonl.txt),
[build and comparison provenance](/benchmark-data/torch-material-runtime-full/comparison-environment.json).
Raw full results and environments:
[before](/benchmark-data/torch-material-runtime-full/before/results.jsonl.txt)
([environment](/benchmark-data/torch-material-runtime-full/before/environment.json)),
[after](/benchmark-data/torch-material-runtime-full/after/results.jsonl.txt)
([environment](/benchmark-data/torch-material-runtime-full/after/environment.json)).

**Native FEM has a measurable cost increase, especially at batch size 1.**
The following are medians over the 20 samples per version from alternating
rechecks, in milliseconds, one thread and two material parameters. Positive
change means slower after migration.

| Native workload | Batch | Before | After | Change |
| --- | ---: | ---: | ---: | ---: |
| FEM value, 1,296 elements | 1 | 0.486 | 0.797 | +64.1% |
| FEM gradient, 1,296 elements | 1 | 0.661 | 1.012 | +53.2% |
| FEM Hessian, 1,296 elements | 1 | 1.910 | 2.281 | +19.4% |
| FEM parameter VJP, 1,296 elements | 1 | 0.560 | 0.738 | +31.7% |
| FEM value, 1,296 elements | 256 | 0.285 | 0.296 | +3.9% |
| FEM gradient, 1,296 elements | 256 | 0.470 | 0.509 | +8.4% |
| FEM Hessian, 1,296 elements | 256 | 1.619 | 1.867 | +15.3% |
| FEM parameter VJP, 1,296 elements | 256 | 0.462 | 0.487 | +5.2% |
| FEM Hessian, 10,368 elements | 256 | 13.265 | 14.091 | +6.2% |
| Equilibrium forward, 48 elements | 256 | 2.226 | 2.339 | +5.1% |
| Equilibrium backward, 48 elements | 256 | 2.674 | 3.031 | +13.3% |

At default batch 256, the single-thread native rechecks increased by about 4–15%
for 1,296-element FEM and 4–10% for 10,368-element FEM. Equilibrium forward and
backward increased by about 5% and 13%. Four-thread results were mixed; a value-only
increase did not imply a similar Hessian increase. These are desktop observations,
not statistical confidence bounds or cross-machine guarantees.

The full-scan median after/before ratios across FEM and equilibrium measurements
were 1.061 for native, 0.947 for Torch and 0.960 for neural. These are unweighted
medians across different workloads, not total application speedups. No Torch
FEM/equilibrium row exceeded a 15% increase. One neural gradient row initially
increased by 24%; alternating rechecks showed a 9% decrease instead, so that
initial apparent regression was not stable.

Native material-only rows are **not** a like-for-like interface cost comparison:
the baseline uses a Python pointwise loop while the new version uses one batch
binding call. They remain in the CSV with `comparable_path=False`, but are excluded
from the FEM/equilibrium summaries above.

The native assembly path now packs batch inputs, stores derivative buffers and
checks batch outputs before integration. These changes are plausible sources of
added overhead, especially for tiny batches; the measurements do not isolate them
from virtual dispatch or other costs. Profiling is needed before attributing the
regression to a specific mechanism. No performance optimization was applied in
this comparison.

## Runtime material interface pilot

Recorded on 2026-11-09 after the runtime batch-material migration, using the
Release build on Apple M1 Pro (8 logical CPUs), macOS 27.0.1, with five measured
samples per operation. The default pilot completed **27 cases and all 98
operation measurements**, with no timeouts or numerical errors. It includes
native, Torch and neural materials, batch-size and thread variations, and neural
weight counts of 2, 8 and 32.

[Raw samples](/benchmark-data/torch-material-runtime-pilot/results.jsonl.txt),
[CSV](/benchmark-data/torch-material-runtime-pilot/timings.csv),
[environment/source hashes](/benchmark-data/torch-material-runtime-pilot/environment.json).

Reproduce with:

```sh
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/torch_material.py --profile pilot --output output/torch_material_runtime_pilot
```

Selected medians in milliseconds, one thread, batch 256, two active parameters:

| Workload | Native | Torch | Neural |
| --- | ---: | ---: | ---: |
| Material Hessian, 32 points | 0.021 | 1.379 | 4.773 |
| FEM Hessian, 48 elements | 0.078 | 1.392 | 5.295 |
| FEM parameter VJP, 48 elements | 0.019 | 1.001 | 2.401 |
| Equilibrium forward, 6 elements | 0.270 | 5.281 | 17.213 |
| Equilibrium backward, 6 elements | 0.416 | 7.539 | 23.991 |

This is a small-mesh pilot, not a repeat of the archived full scan. These results
do not establish a migration speedup or slowdown: the historical tables use
larger workloads, and the native material timing path changed from a Python
pointwise loop to a batch adapter. The neural formula also differs from native
and Torch Neo-Hookean. All runs are desktop observations with the measurement
boundaries above.

## Historical stateless batched implementation

The current implementation uses explicit tensor weights, batched Linear/Softplus operations, and cached reference features/slopes for the fixed hidden layers. It never replaces module parameters and needs no lock. The benchmark and fitting example share `make_material(K)`, supporting 1–32 active final-layer weights. Hidden weights remain fixed; this is not an implementation for training all network layers.

The pre-migration full scan completed **42 cases and all 152 operation measurements**, with no timeouts or numerical errors. Tests additionally compare energy, stress, Hessian, weight mixed derivatives, and plastic mixed derivatives against the old functional-call formula for distinct parameter rows at K=2/8/32.

[Archived raw samples](/benchmark-data/torch-material-batched/results.jsonl.txt), [CSV](/benchmark-data/torch-material-batched/timings.csv), [environment/source hashes](/benchmark-data/torch-material-batched/environment.json).

Same machine and settings as the historical baseline; separate runs, not paired randomized experiments. Medians in milliseconds, one thread, batch 256, two active weights:

| Workload | Pointwise | Stateless batch | Observed ratio |
| --- | ---: | ---: | ---: |
| FEM Hessian, 1,296 elements | 4001.090 | 44.561 | 89.8x |
| FEM parameter VJP, 1,296 elements | 2890.023 | 19.716 | 146.6x |
| Equilibrium forward, 48 elements | 715.241 | 29.458 | 24.3x |
| Equilibrium backward, 48 elements | 939.663 | 34.676 | 27.1x |

The change combines batching, removal of temporary module replacement, and algebraically moving fixed hidden-layer reference corrections out of repeated evaluation. These measurements do not isolate the contribution of each change. The older profiler and all older measurements below are retained as historical evidence.

## Historical pointwise baseline

The results below are a local baseline, not a cross-machine performance promise.
Large cases that reach the time budget remain explicit incomplete observations.

Recorded on 2026-11-07: **Apple M1 Pro (8 logical CPUs), macOS-27.0.1-arm64-arm-64bit**, Release build, Python 3.12, Torch 2.15.0, NumPy 2.5.4. The full profile ran 42 cases with five measured samples per completed operation. There were **149 validated operation records, two partial-case timeouts, and no numerical errors**.

[Raw samples (JSONL)](/benchmark-data/torch-material-baseline/results.jsonl.txt), [summary CSV](/benchmark-data/torch-material-baseline/timings.csv), [environment and source hashes](/benchmark-data/torch-material-baseline/environment.json), [targeted profile](/benchmark-data/torch-material-baseline/neural_profile.txt).

The archived JSONL has an extra `.txt` suffix for static hosting; its contents are unchanged.

### Assembled FEM: 1,296 elements

All times below are medians in milliseconds, one thread, batch 256. Neural uses two active weights.

| Material | Value | Gradient | Exact Hessian | Parameter VJP |
| --- | ---: | ---: | ---: | ---: |
| native | 0.280 | 0.435 | 1.535 | 0.443 |
| torch | 1.077 | 1.835 | 10.729 | 7.468 |
| neural | 239.638 | 436.644 | 4001.090 | 2890.023 |

For the same Neo-Hookean formula, Torch Hessian evaluation is about 7.0 times the native time here; parameter VJP is about 16.9 times. The neural formula is different, and its VJP differentiates network weights, so its ratios are workload costs rather than implementation-only slowdowns.

### Mesh and batch scaling

| Elements | Native Hessian (ms) | Torch Hessian (ms) | Neural Hessian (ms) |
| ---: | ---: | ---: | ---: |
| 162 | 0.216 | 1.638 | 481.150 |
| 1,296 | 1.535 | 10.729 | 4001.090 |
| 10,368 | 12.429 | 78.283 | — |

At 1,296 elements and one thread:

| Batch elements | Torch Hessian (ms) | Neural Hessian (ms) | Neural parameter VJP (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 1389.207 | 4988.561 | 2988.357 |
| 32 | 48.311 | 3593.344 | 2534.886 |
| 256 | 10.729 | 4001.090 | 2890.023 |
| 1024 | 6.394 | 4493.712 | 3252.336 |

Batch 32 was faster than 256 for this pointwise neural implementation; bigger batches are not uniformly better. At batch 256, changing both Torch/TBB settings from 1 to 4 threads gave:

| Material | Hessian, 1 thread (ms) | Hessian, 4 threads (ms) | VJP, 1 thread (ms) | VJP, 4 threads (ms) |
| --- | ---: | ---: | ---: | ---: |
| native | 1.535 | 0.541 | 0.443 | 0.190 |
| torch | 10.729 | 6.249 | 7.468 | 5.367 |
| neural | 4001.090 | 1988.008 | 2890.023 | 1454.985 |

The protected neural forward calls do not prevent graph differentiation/native assembly from benefiting from concurrency. These measurements do not isolate Torch threads from TBB threads.

### Active final-layer weights

| Weights | Material mixed derivative, 256 points (ms) | FEM parameter VJP, 162 elements (ms) | Equilibrium backward, 6 elements (ms) |
| ---: | ---: | ---: | ---: |
| 2 | 566.590 | 348.080 | 101.174 |
| 8 | 565.258 | 342.331 | 100.353 |
| 32 | 574.912 | 348.257 | 104.577 |

These small final-layer weight counts reuse the same network graph and require the same nine reverse passes for mixed derivatives. They do not measure full-network training or establish that cost is independent of parameter count.

### Equilibrium: 48 elements

| Material | Forward (ms) | Loss backward (ms) | Diagnostic Newton iterations |
| --- | ---: | ---: | ---: |
| native | 1.957 | 2.275 | 4 |
| torch | 8.457 | 10.934 | 4 |
| neural | 715.241 | 939.663 | 4 |

All three workloads converged in four iterations. Backward includes another solve: the forward and backward columns are separate costs, not two phases of a single solve. A phase-level native profiler would be needed to attribute the remaining backward time precisely.

### Memory and time limits

| 1,296-element case, batch 256, 1 thread | Post-import peak (MiB) | Case high-water mark (MiB) |
| --- | ---: | ---: |
| native | 207.1 | 227.8 |
| torch | 206.8 | 239.3 |
| neural | 207.0 | 279.4 |

The two timeouts were **cumulative 90-second case budgets**, including setup, validation, warmups and repeats:

- Neural material at 2,048 points completed value, stress, and Hessian samples; its parameter mixed-derivative timing is missing.
- Neural FEM at 10,368 elements completed value and gradient samples; Hessian and VJP timings are missing.

A timeout does not mean one operator call took 90 seconds. Partial rows remain in the raw results. The reported RSS is a process high-water mark including validation, not an isolated allocation benchmark.

### Original optimization hypothesis

Start with the neural material evaluation and derivative graph. Even without assembly, the 256-point neural Hessian takes about 789 ms. The pointwise implementation builds many separate network/reference-correction subgraphs and traverses them for second derivatives. A targeted instrumented profile of the fitting example is included above; it is diagnostic evidence, not a replacement for the uninstrumented samples.

A batched implementation that preserves independent parameter rows is the first experiment to compare against this baseline. Keep the lock until shared-module replacement is removed or made independently safe. Full-network direct mixed-derivative products, GPU execution, compilation, and cached equilibrium linearizations remain separate experiments; none was implemented for this benchmark.

The targeted profile spent about 0.767 of 0.831 instrumented seconds in the
autograd engine. It recorded 266 gradient calls: 256 reference-slope evaluations
inside the pointwise network, plus one stress and nine Hessian-row evaluations.
This supports reducing the size and fragmentation of the differentiated graph;
it does not support blaming the whole cost on Python call overhead or data copies.
