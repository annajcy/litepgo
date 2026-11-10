# Energy evaluation input sharing

Historical measurements: input sharing was later removed in favor of uniform
input ownership. See [the owning-input comparison](energy-own.md).


The historical allocation/input-sharing optimization was implemented on top of
native cache reuse (`7b17320`). It replaced the earlier experiment 2 prototype;
the small-workload serial-path experiment remained disabled.

## Historical implementation and API

Public C++/Python `evaluate(x)` took an independent snapshot. Internal
WeightedSum children borrowed their parent's read-only input, including nested
sums. Affine-mapped child evaluations owned `T*x+b`. Default leaf states used
`make_shared` to combine object/control-block allocation. The Energy model was
borrowed, with no additional shared model ownership.

Evaluation objects could not be copied or moved, and their input storage/view
was const. Built-in child states were private and were destroyed before their
parent's snapshot. The internal child-creation helper took a parent evaluation
rather than an arbitrary vector.

That version required custom C++ factories to accept an input-ownership flag,
and exposed protected `x_` as a read-only `ConstVecRef`. This contract has been
removed: current factories take only `make_evaluation(x)`, and every evaluation
owns its input in a read-only `Vec`. See the current
[custom evaluation contract](../guide/energy/custom.md).

## Correctness

Release CTest: 509/509, including Python 660 passed / 81 skipped. ASan/UBSan:
88 Energy/Response and 35 solver tests passed. The native regression at that revision verified
input pointer sharing, distinct live public snapshots, custom-factory policy
forwarding, interleaved evaluation destruction, exact/projected derivatives,
feasible steps, and reading the input during child destruction. Python tests at that revision
exercised nested weighted/affine states with a non-contiguous input that is
mutated and deleted, model garbage collection, and both projection orders.
All examples build. Tutorial notebooks were not rerun because their APIs did not
change; the historical factory contract is described above.

## Paired performance measurements

The reference is the committed cache-only implementation, not the pre-migration
API. Both versions use frozen modules with copied Python wrappers. The original
146 Energy/equilibrium/control cases plus 36 composite cases run in
before/after/after/before order, with no concurrent builds or tests. Each version
has two process runs: 30 Energy samples/case, 14 equilibrium samples/case,
40 Response-control samples/case, and 42 composite samples/case. Composite cases
warm up 50 calls and time 100 calls/sample for creation/gradient, 20 for Hessian.
All values, gradients, Hessian actions and equilibrium states agree across builds
at `rtol=2e-8`, `atol=1e-9`. Each equilibrium operation still solves once.

Representative composite results (1 thread):

| Case | Before μs | After μs | Change |
| --- | ---: | ---: | ---: |
| leaf/3/1/create | 0.355 | 0.369 | +4.2% |
| nested/3/1/create | 0.878 | 0.702 | -20.1% |
| nested/1000/1/create | 3.519 | 2.355 | -33.1% |
| nested/10000/1/create | 25.078 | 14.978 | -40.3% |
| nested/10000/1/gradient | 212.055 | 201.314 | -5.1% |
| nested/10000/1/hessian | 451.476 | 443.218 | -1.8% |
| mapped/10000/1/gradient | 134.034 | 130.127 | -2.9% |

Equilibrium geometric means of per-case ratios (24 cases per family):

| Family | Time change |
| --- | ---: |
| tet | -2.0% |
| hermite | -0.3% |
| shell | -0.7% |
| shell_fp | -3.0% |

The optimization targets input/state setup, so benefits depend on composition
depth and input size. It does not promise a speedup for every leaf or an end-to-end
FEM workload dominated by assembly and factorization. Small multithreaded timing
differences remain sensitive to scheduling; the CSVs include every case, including
regressions. This comparison does not isolate each allocation's cost or measure
peak live allocation counts.

## Reproduction and data

The local frozen modules are `build/energy-input-optimization/before-python` and
`after-python`; use PYTHONPATH to select one per process. Fingerprints, comparison
CSVs, exact drivers, and the production source patch are retained below.

- [environment.json](/benchmark-data/energy-input-optimization/environment.json)
- [energy.csv](/benchmark-data/energy-input-optimization/energy.csv)
- [equilibrium.csv](/benchmark-data/energy-input-optimization/equilibrium.csv)
- [composites.csv](/benchmark-data/energy-input-optimization/composites.csv)
- [run.sh.txt](/benchmark-data/energy-input-optimization/run.sh.txt)
- [energy_evaluation.py.txt](/benchmark-data/energy-input-optimization/energy_evaluation.py.txt)
- [equilibrium_evaluation.py.txt](/benchmark-data/energy-input-optimization/equilibrium_evaluation.py.txt)
- [energy_cache.py.txt](/benchmark-data/energy-input-optimization/energy_cache.py.txt)
- [implementation.patch.txt](/benchmark-data/energy-input-optimization/implementation.patch.txt)

<a href="/benchmark-data/energy-input-optimization/raw-samples.tar.xz" download>All raw samples</a> (lossless tar.xz containing the original JSONL numerical outputs and timings).

Process resource measurements:

- [0-before-composites.resources.txt](/benchmark-data/energy-input-optimization/0-before-composites.resources.txt)
- [0-before-energy.resources.txt](/benchmark-data/energy-input-optimization/0-before-energy.resources.txt)
- [0-before-equilibrium.resources.txt](/benchmark-data/energy-input-optimization/0-before-equilibrium.resources.txt)
- [1-after-composites.resources.txt](/benchmark-data/energy-input-optimization/1-after-composites.resources.txt)
- [1-after-energy.resources.txt](/benchmark-data/energy-input-optimization/1-after-energy.resources.txt)
- [1-after-equilibrium.resources.txt](/benchmark-data/energy-input-optimization/1-after-equilibrium.resources.txt)
- [2-after-composites.resources.txt](/benchmark-data/energy-input-optimization/2-after-composites.resources.txt)
- [2-after-energy.resources.txt](/benchmark-data/energy-input-optimization/2-after-energy.resources.txt)
- [2-after-equilibrium.resources.txt](/benchmark-data/energy-input-optimization/2-after-equilibrium.resources.txt)
- [3-before-composites.resources.txt](/benchmark-data/energy-input-optimization/3-before-composites.resources.txt)
- [3-before-energy.resources.txt](/benchmark-data/energy-input-optimization/3-before-energy.resources.txt)
- [3-before-equilibrium.resources.txt](/benchmark-data/energy-input-optimization/3-before-equilibrium.resources.txt)
