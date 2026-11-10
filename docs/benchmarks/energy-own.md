# Uniformly owned evaluation inputs

Every EnergyEvaluation now owns an immutable input vector, including composite
children. The input-ownership flag, internal child-creation helper, and separate
owned-storage/view members are removed. The C++ factory is make_evaluation(x).
Energy models remain borrowed. Native derivative-cache reads, make_shared leaf
allocation, child-vector reservation and the exact-equilibrium matrix shortcut
are all retained. Python numerical APIs are unchanged.

This is a deliberate simplicity tradeoff: nested evaluations copy their inputs
again. Historical sharing experiments remain archived, not active API examples.
The ownership regression checks distinct live child buffers as well as caller
mutation, interleaved state destruction, derivative caches and feasible steps.
Existing Python tests cover nested affine/weighted states and non-contiguous
inputs. Release CTest passed 509/509 (Python 662 passed, 81 skipped), and
ASan/UBSan passed 88 Energy/Response and 35 solver tests. All examples build.
Tutorial notebooks were not rerun because their public API is unchanged.

## Paired comparison

Before is commit 4f4d13b with shared child inputs; after uses owned inputs.
The original 146 cases plus 36 focused composite cases ran in before/after/after/before
order, with isolated modules, the same Release configuration and no concurrent
builds/tests. Each version pools 30 Energy samples, 14 equilibrium samples,
40 Response-control samples and 42 composite samples per case. All numerical
outputs agree (rtol=2e-8, atol=1e-9); each equilibrium operation solves once.
Positive percentages mean the owning version took longer.

| Case | Before μs | After μs | Change |
| --- | ---: | ---: | ---: |
| leaf/3/1/create | 0.370 | 0.387 | +4.6% |
| nested/3/1/create | 0.642 | 0.812 | +26.5% |
| nested/10000/1/create | 15.212 | 25.896 | +70.2% |
| nested/10000/1/gradient | 206.972 | 220.276 | +6.4% |
| nested/10000/1/hessian | 471.061 | 475.358 | +0.9% |

Equilibrium geometric mean of time ratios (24 cases/family):

| Family | Change |
| --- | ---: |
| tet | -1.3% |
| hermite | +0.9% |
| shell | -3.3% |
| shell_fp | -0.6% |

These workloads do not establish universal speedups or regressions. Small multithreaded differences remain sensitive to scheduling. Full per-case results, including regressions, follow.

- [environment.json](/benchmark-data/energy-own/environment.json)
- [energy.csv](/benchmark-data/energy-own/energy.csv)
- [equilibrium.csv](/benchmark-data/energy-own/equilibrium.csv)
- [composites.csv](/benchmark-data/energy-own/composites.csv)
- [run.sh.txt](/benchmark-data/energy-own/run.sh.txt)
- [energy_evaluation.py.txt](/benchmark-data/energy-own/energy_evaluation.py.txt)
- [equilibrium_evaluation.py.txt](/benchmark-data/energy-own/equilibrium_evaluation.py.txt)
- [energy_cache.py.txt](/benchmark-data/energy-own/energy_cache.py.txt)
- [implementation.patch.txt](/benchmark-data/energy-own/implementation.patch.txt)

<a href="/benchmark-data/energy-own/raw-samples.tar.xz" download>Raw numerical outputs and timing samples</a> (lossless tar.xz).
