# Energy evaluation input sharing

The allocation/input-sharing optimization is now implemented on top of native
cache reuse (`7b17320`). It replaces the earlier experiment 2 prototype; the
small-workload serial-path experiment remains disabled.

## Implementation and API

Public C++/Python `evaluate(x)` still takes an independent snapshot. Internal
WeightedSum children borrow their parent's read-only input; this also applies to
nested sums. Affine-mapped child evaluations own `T*x+b`. Default leaf states use
`make_shared` to combine object/control-block allocation. The Energy model is
still borrowed, with no additional shared model ownership.

Evaluation objects cannot be copied or moved, and their input storage/view are
const. Built-in child states are private and are destroyed before their parent's
snapshot. The internal child-creation helper takes a parent evaluation rather
than an arbitrary vector.

Advanced C++ subclasses must override `make_evaluation(x, bool borrow_input)` and
forward the flag to the state constructor, or choose to copy the input. The
protected `x_` is now a read-only ConstVecRef. Public evaluate calls pass false;
borrowed children must not escape their parent. The ordinary numerical hooks,
public C++/Python methods, and independently owned derivative results are
unchanged. See the [custom evaluation contract](../guide/energy/custom.md).

## Correctness

Release CTest: 509/509, including Python 660 passed / 81 skipped. ASan/UBSan:
88 Energy/Response and 35 solver tests passed. The native regression verifies
input pointer sharing, distinct live public snapshots, custom-factory policy
forwarding, interleaved evaluation destruction, exact/projected derivatives,
feasible steps, and reading the input during child destruction. Python tests
exercise nested weighted/affine states with a non-contiguous input that is
mutated and deleted, model garbage collection, and both projection orders.
All examples build. Tutorial notebooks were not rerun because their APIs did not
change; the advanced C++ factory migration is documented separately.

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
