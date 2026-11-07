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

At the material layer, native Neo-Hookean is available only through **pointwise
Python bindings**, so those rows include a Python loop. They are not a pure C++
batched throughput baseline. The native FEM rows provide that compiled assembly
baseline. Torch material calls return just the requested output; computing a
second derivative still requires the first derivative internally.

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

## Stateless batched implementation

The current implementation uses explicit tensor weights, batched Linear/Softplus operations, and cached reference features/slopes for the fixed hidden layers. It never replaces module parameters and needs no lock. The benchmark and fitting example share `make_material(K)`, supporting 1–32 active final-layer weights. Hidden weights remain fixed; this is not an implementation for training all network layers.

The updated full scan completed **42 cases and all 152 operation measurements**, with no timeouts or numerical errors. Tests additionally compare energy, stress, Hessian, weight mixed derivatives, and plastic mixed derivatives against the old functional-call formula for distinct parameter rows at K=2/8/32.

[New raw samples](/benchmark-data/torch-material-batched/results.jsonl.txt), [CSV](/benchmark-data/torch-material-batched/timings.csv), [environment/source hashes](/benchmark-data/torch-material-batched/environment.json).

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
