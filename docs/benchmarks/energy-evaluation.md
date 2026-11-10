# Energy evaluation migration performance

Measured on Apple M1 Pro (16 GiB), macOS, Release `-O3 -DNDEBUG`, arm64.
The migration replaces scalar tuple evaluation and `HessianKind` with lazy
`EnergyEvaluation` objects and separate exact/projected Hessian methods. Response
C++ evaluations now borrow their model; Python bindings retain it.

This is an API/state migration, **not a general speedup**. Small calls incur
allocation, input-copy and Python-call overhead. Exact and projected caches keep
independent results, and Hessian requests also prepare gradient/value. Existing
FEM/contact local projection paths are retained.

## Conditions and validation

The baseline is commit `d3898cf` plus the Response borrowing-only change, before
any Energy migration. Both native modules use the same Release build configuration.
Python wrappers and the old-API drivers were frozen alongside the baseline binary.
No builds, tests or tutorial execution overlapped the measurements.

Four fresh-process runs use before → after → after → before order. Each version
has 30 timing samples per Energy case and 14 per equilibrium case. Results below
use pooled medians; positive percentages mean longer time. The complete matrix
has **146 cases**: 48 Energy cases, 96 equilibrium cases and two tiny Response
controls. The controls pool 40 samples per version.

- Energy: 3/1000-DOF diagonal quadratic models; TetP1 (48 elements, 81 DOFs),
  HexHermite (8 elements, 648 DOFs), shell (8 faces, 27 DOFs). FEM uses 1/4 TBB threads.
- Operations: value, value/gradient, joint value/gradient/exact Hessian, joint
  projected Hessian, and exact/projected Hessian-only calls. Joint requests ask
  Hessian first under the new API; Hessian-only requests include the new contract's
  implicit gradient/value work.
- Equilibrium: TetP1, HexHermite, shell theta/Fp; two mesh sizes; exact/projected
  Newton; standalone VJP, Torch forward/backward and composed forward/backward.
- Values, gradients, two deterministic Hessian actions and equilibrium states
  agree across versions (relative tolerance `2e-8`, absolute `1e-9`). Each driver
  also checks repeatability; equilibrium checks residuals and small-case finite
  differences. All 96 equilibrium cases use **one nonlinear solve** per operation
  before and after; residuals stay at or below `1.1e-9`.

## Small scalar calls

Times are microseconds. Even a fraction of a microsecond produces a large
percentage for the smallest value-only call.

| DOFs | Operation | Before μs | After μs | Change |
| --- | --- | ---: | ---: | ---: |
| 3 | value | 0.270 | 0.486 | +80.1% |
| 3 | gradient | 0.981 | 1.190 | +21.3% |
| 3 | exact | 8.138 | 8.448 | +3.8% |
| 3 | projected | 8.132 | 8.408 | +3.4% |
| 3 | hessian_only | 7.374 | 7.775 | +5.4% |
| 3 | projected_only | 7.362 | 7.593 | +3.1% |
| 1000 | value | 1.730 | 2.136 | +23.4% |
| 1000 | gradient | 2.653 | 3.128 | +17.9% |
| 1000 | exact | 11.160 | 12.812 | +14.8% |
| 1000 | projected | 11.436 | 12.701 | +11.1% |
| 1000 | hessian_only | 10.409 | 11.829 | +13.6% |
| 1000 | projected_only | 10.405 | 11.801 | +13.4% |

## FEM joint derivatives

These request the same three outputs in both versions. Local Hessian projection
is preserved, so the benchmark does not substitute global dense projection for FEM.

| Family | Threads | Hessian | Before μs | After μs | Change |
| --- | ---: | --- | ---: | ---: | ---: |
| tet | 1 | exact | 81.78 | 83.91 | +2.6% |
| tet | 1 | projected | 311.95 | 314.61 | +0.9% |
| tet | 4 | exact | 83.70 | 84.48 | +0.9% |
| tet | 4 | projected | 320.27 | 325.82 | +1.7% |
| hermite | 1 | exact | 2747.69 | 2826.18 | +2.9% |
| hermite | 1 | projected | 24242.82 | 24447.41 | +0.8% |
| hermite | 4 | exact | 1244.94 | 1351.87 | +8.6% |
| hermite | 4 | projected | 22589.07 | 22875.03 | +1.3% |
| shell | 1 | exact | 40.96 | 40.97 | +0.0% |
| shell | 1 | projected | 94.21 | 95.10 | +0.9% |
| shell | 4 | exact | 41.81 | 46.27 | +10.7% |
| shell | 4 | projected | 72.39 | 75.30 | +4.0% |

The CSV also includes value, gradient and Hessian-only FEM cases. Across all six
operations and both thread counts, duration-ratio geometric means are:

| Family | Geometric mean change | Case range |
| --- | ---: | ---: |
| tet | +5.0% | +0.9% to +20.0% |
| hermite | +3.6% | +0.8% to +10.7% |
| shell | +9.0% | -4.7% to +45.9% |

## End-to-end equilibrium

Each row aggregates 24 cases: two sizes, two thread counts, two Hessian choices
and three derivative paths. These small synthetic meshes do not predict large
production solves.

| Family | Geometric mean change | Case range |
| --- | ---: | ---: |
| tet | +3.0% | -8.8% to +44.6% |
| hermite | +2.4% | -4.9% to +7.5% |
| shell | +2.3% | -17.8% to +17.9% |
| shell_fp | +1.3% | -6.9% to +19.8% |

Small individual timings vary materially between fresh processes. Two conspicuous
cases were rechecked with 50 warmup calls, then 31 samples of 20 calls per process,
again before → after → after → before (62 samples per version):

| Recheck | Before μs | After μs | Change |
| --- | ---: | ---: | ---: |
| tet_torch | 241.26 | 236.54 | -2.0% |
| shell_gradient | 15.66 | 20.43 | +30.5% |

The large initial Tet/Torch outlier does not persist with longer warmup. The
four-thread tiny-shell gradient slowdown persisted in that recheck. A later,
longer [profiling recheck](energy-evaluation-profiling.md) changes the sign of this
difference, so +30.5% must not be treated as a stable migration cost. There is no claim of zero regression. The data establishes
API overhead and small-workload costs, not a profiler-based attribution of every
microsecond. Raw original results remain included alongside the recheck.

## Scope and reproduction

The new PSD contract intentionally changes indefinite `QuadraticEnergy` and
negative-weight sums; correctness tests cover those changes. Performance cases
use comparable mathematical problems. Default spectral projection may densify a
generic Hessian; it is not suitable as the default path for a large sparse FEM
problem. Native FEM/contact retain local projection. Independent caches can also
increase live memory; process-level RSS files include Python/Torch and are not a
measurement of cache storage alone.

```sh
PYTHONPATH=build/release/python .venv/bin/python benchmarks/energy_evaluation.py --samples 15
PYTHONPATH=build/release/python .venv/bin/python benchmarks/equilibrium_evaluation.py --samples 7
```

The archived old drivers require the captured pre-migration module; they are
historical evidence, not supported old-API examples. Do not run timings alongside
builds or other benchmarks.

- [All-case comparison CSV](/benchmark-data/energy-evaluation/comparison.csv)
- [Environment and binary fingerprints](/benchmark-data/energy-evaluation/environment.json)
- [Old Energy driver](/benchmark-data/energy-evaluation/energy_before.py.txt)
- [New Energy driver](/benchmark-data/energy-evaluation/energy_evaluation.py.txt)
- [Old equilibrium driver](/benchmark-data/energy-evaluation/equilibrium_before.py.txt)
- [New equilibrium driver](/benchmark-data/energy-evaluation/equilibrium_evaluation.py.txt)
- [Comparison validator](/benchmark-data/energy-evaluation/summarize.py.txt)
- [Supplemental recheck driver](/benchmark-data/energy-evaluation/recheck.py.txt)
- [Run commands](/benchmark-data/energy-evaluation/run_comparison.sh.txt)

Raw samples and process resources:

| Run | Energy | Equilibrium | Resources |
| --- | --- | --- | --- |
| 0: before | [samples](/benchmark-data/energy-evaluation/final-0-energy-before.jsonl.txt) | [samples](/benchmark-data/energy-evaluation/final-0-equilibrium-before.jsonl.txt) | [Energy](/benchmark-data/energy-evaluation/final-0-energy-before.resources.txt), [equilibrium](/benchmark-data/energy-evaluation/final-0-equilibrium-before.resources.txt) |
| 1: after | [samples](/benchmark-data/energy-evaluation/final-1-energy-after.jsonl.txt) | [samples](/benchmark-data/energy-evaluation/final-1-equilibrium-after.jsonl.txt) | [Energy](/benchmark-data/energy-evaluation/final-1-energy-after.resources.txt), [equilibrium](/benchmark-data/energy-evaluation/final-1-equilibrium-after.resources.txt) |
| 2: after | [samples](/benchmark-data/energy-evaluation/final-2-energy-after.jsonl.txt) | [samples](/benchmark-data/energy-evaluation/final-2-equilibrium-after.jsonl.txt) | [Energy](/benchmark-data/energy-evaluation/final-2-energy-after.resources.txt), [equilibrium](/benchmark-data/energy-evaluation/final-2-equilibrium-after.resources.txt) |
| 3: before | [samples](/benchmark-data/energy-evaluation/final-3-energy-before.jsonl.txt) | [samples](/benchmark-data/energy-evaluation/final-3-equilibrium-before.jsonl.txt) | [Energy](/benchmark-data/energy-evaluation/final-3-energy-before.resources.txt), [equilibrium](/benchmark-data/energy-evaluation/final-3-equilibrium-before.resources.txt) |

Supplemental samples: [0](/benchmark-data/energy-evaluation/recheck-0-before.jsonl.txt), [1](/benchmark-data/energy-evaluation/recheck-1-after.jsonl.txt), [2](/benchmark-data/energy-evaluation/recheck-2-after.jsonl.txt), [3](/benchmark-data/energy-evaluation/recheck-3-before.jsonl.txt).

See the [profiling follow-up and corrected baseline](energy-evaluation-profiling.md) for the Debug fixes and the baseline to use for subsequent optimization.
