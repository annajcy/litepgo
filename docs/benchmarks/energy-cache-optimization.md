# Energy cache optimization and allocation/serial-path experiments

Experiment 2 was subsequently implemented and validated separately; see
[the input-sharing follow-up](energy-input-optimization.md). The results below
describe the earlier cache-only implementation and experimental prototypes.

The existing workspace was committed as `fbc5449` before this work. Only the
native derivative-cache optimization is retained in the working tree. The two
other prototypes were built, checked, timed independently, and removed from the
production source. They remain available as patches and isolated local modules.

## Retained change: native cache reads

Public C++ and Python gradient/Hessian methods still return independently owned
results. Internal weighted/affine composition, Newton and equilibrium backward
read validated cached derivatives by const reference. The default spectral
projection also reads the exact Hessian directly. Gradient caches keep their
storage when a later derivative request supplies another joint result.

Newton compares the next Hessian against the full input pattern already stored
by ReducedMatrix. It no longer copies a complete Hessian just to retain the
previous iteration's pattern. Changing-pattern and projected/exact behavior
remain covered by solver tests.

Validation: Release CTest 508/508, including Python 658 passed / 81 skipped;
Debug and ASan/UBSan each passed 87 Energy/Response, 35 solver and 56 DOF tests.
The new native test holds cache references across exact/projected upgrades and
other evaluations and checks that public owning results remain independent.
After restoring the production source from the prototypes, its bytes were
checked against the tested patch and the Release build and related tests were
rerun. No public Energy API migration is needed in examples or tutorials.

## Measurements against the corrected baseline

The reference is the frozen module from the [corrected baseline](energy-evaluation-profiling.md),
not the earlier pre-migration API. The original 146 cases plus 36 focused
composite cases ran in before/after/after/before order, with the same Release
configuration and 1/4 TBB threads. Two processes per version pool 30 Energy
samples, 14 equilibrium samples, 40 tiny Response-control samples, and 42 focused
samples. Focused cases warm up 50 calls and time 100 calls/sample for creation or
gradient and 20 calls/sample for Hessians. All numerical outputs agree across
versions (`rtol=2e-8`, `atol=1e-9`); each equilibrium operation solves once.

Representative 10000-DOF tridiagonal composite Hessians:

| Case | Before μs | After μs | Change |
| --- | ---: | ---: | ---: |
| leaf/10000/1/hessian | 77.841 | 77.533 | -0.4% |
| pair/10000/1/hessian | 223.867 | 198.425 | -11.4% |
| nested/10000/1/hessian | 509.075 | 447.730 | -12.1% |
| mapped/10000/1/hessian | 304.053 | 271.282 | -10.8% |

Equilibrium geometric means of per-case ratios (24 cases per family):

| Family | Time change |
| --- | ---: |
| tet | -1.1% |
| hermite | -1.2% |
| shell | -4.7% |
| shell_fp | -2.1% |

These small end-to-end percentages should not be interpreted as universal
speedups. Tiny unchanged Response controls also shift by a few percent. Some
four-thread FEM cases move in the other direction; all cases are in the CSVs.
The substantial composite-Hessian reductions are clearer than the small-mesh
aggregate effects. Constitutive work and Python sparse conversion still dominate
many other cases.

The initial 10000-DOF mapped gradient showed +4.8%. A longer independent ABBA
recheck uses 41 samples × 300 calls per process:

| Case | Before μs | After μs | Change |
| --- | ---: | ---: | ---: |
| leaf/10000/1/gradient | 46.615 | 46.635 | +0.0% |
| pair/10000/1/gradient | 104.020 | 104.724 | +0.7% |
| mapped/10000/1/gradient | 138.053 | 133.676 | -3.2% |

The earlier mapped-gradient slowdown did not reproduce in this longer run.

## Experiment 2: allocation and input snapshots

On top of the retained change, the prototype combines the leaf evaluation and
shared-pointer control block in one allocation. WeightedSum children borrow the
parent evaluation's input snapshot; a public evaluate call still copies its
input. Affine-mapped children own their transformed input. Energy model ownership
is unchanged. The experiment changes protected C++ evaluation factory signatures,
so it is not a drop-in production patch for external custom evaluation classes.
Its eventual factory design needs to preserve or deliberately document that
extension contract before adoption.

Related Release tests and 42 Python tests pass; 87 Energy/Response and 35 solver
ASan/UBSan tests pass. Measurements compare against the retained cache change,
with the same ABBA ordering:

| Case | Before μs | After μs | Change |
| --- | ---: | ---: | ---: |
| leaf/3/1/create | 0.354 | 0.375 | +5.9% |
| nested/10000/1/create | 25.370 | 14.983 | -40.9% |
| nested/10000/1/gradient | 219.680 | 200.315 | -8.8% |
| nested/10000/1/hessian | 453.431 | 441.318 | -2.7% |
| mapped/10000/1/hessian | 274.183 | 270.660 | -1.3% |

## Experiment 3: small-workload serial execution

A separate prototype validates at most eight element views serially and directly
calls batch evaluation when exactly one batch exists. All dimension/finiteness
checks and material computations remain. It does not include experiment 2.
Related Release tests and 54 Python tests pass.

The count threshold is an experimental choice, not a tuned general policy:
eight Hermite element Hessians contain far more data than eight shell gradients.
The timing sweep includes shell sizes 1/2/4/8 and TetP1/HexHermite sizes 1/2 with
1/4 threads. Small-shell and Hermite cases illustrate that distinction:

| Case | Before μs | After μs | Change |
| --- | ---: | ---: | ---: |
| shell/1/4/gradient | 4.504 | 4.018 | -10.8% |
| shell/2/4/gradient | 13.257 | 12.862 | -3.0% |
| shell/4/4/gradient | 30.304 | 33.458 | +10.4% |
| shell/8/4/gradient | 85.082 | 85.289 | +0.2% |
| hermite/2/4/hessian | 1232.766 | 1283.640 | +4.1% |
| tet/1/4/hessian | 23.177 | 23.038 | -0.6% |

Experimental equilibrium aggregates (each compared with its own cache-only control runs):

| Family | Allocation/input prototype | Serial-path prototype |
| --- | ---: | ---: |
| tet | +1.7% | -0.4% |
| hermite | -0.4% | -0.3% |
| shell | +2.1% | +1.2% |
| shell_fp | +5.5% | +0.1% |

Neither experimental patch is enabled in the final build. Small differences need
repetition on representative application meshes before choosing a production
threshold or accepting a more complex lifetime/factory design. These experiments
do not measure peak live allocations or establish scaling beyond the tested sizes.

The retained cache optimization has the clearest benefit for composite Hessians.
Experiment 2 is promising for deeply nested or large-input compositions, but its
small equilibrium workloads show no consistent improvement. Experiment 3 helps
the smallest gradient calls without establishing an end-to-end benefit; a general
serial threshold should consider derivative work, not only element count.

## Reproduction and data

Run benchmarks only after builds/tests have finished. Use PYTHONPATH to select
one isolated module per process; the environment file records binary and source
fingerprints. Local modules are under `build/energy-cache-optimization/`:
`cache-python`, `experiment2-python`, and `experiment3-python`. Their Python
wrappers are copied, not symlinked.

- [cache-energy comparison](/benchmark-data/energy-cache-optimization/cache-energy.csv)
- [cache-equilibrium comparison](/benchmark-data/energy-cache-optimization/cache-equilibrium.csv)
- [cache-composites comparison](/benchmark-data/energy-cache-optimization/cache-composites.csv)
- [experiment2-composites comparison](/benchmark-data/energy-cache-optimization/experiment2-composites.csv)
- [experiment2-equilibrium comparison](/benchmark-data/energy-cache-optimization/experiment2-equilibrium.csv)
- [experiment3-assembly comparison](/benchmark-data/energy-cache-optimization/experiment3-assembly.csv)
- [experiment3-equilibrium comparison](/benchmark-data/energy-cache-optimization/experiment3-equilibrium.csv)
- [gradient-recheck comparison](/benchmark-data/energy-cache-optimization/gradient-recheck.csv)
- [environment.json](/benchmark-data/energy-cache-optimization/environment.json)
- [cache.patch.txt](/benchmark-data/energy-cache-optimization/cache.patch.txt)
- [experiment2.patch.txt](/benchmark-data/energy-cache-optimization/experiment2.patch.txt)
- [experiment3.patch.txt](/benchmark-data/energy-cache-optimization/experiment3.patch.txt)
- [energy_cache.py.txt](/benchmark-data/energy-cache-optimization/energy_cache.py.txt)
- [energy_evaluation.py.txt](/benchmark-data/energy-cache-optimization/energy_evaluation.py.txt)
- [equilibrium_evaluation.py.txt](/benchmark-data/energy-cache-optimization/equilibrium_evaluation.py.txt)
- [run_step1.sh.txt](/benchmark-data/energy-cache-optimization/run_step1.sh.txt)
- [run_experiments.sh.txt](/benchmark-data/energy-cache-optimization/run_experiments.sh.txt)
- [run_recheck.sh.txt](/benchmark-data/energy-cache-optimization/run_recheck.sh.txt)
- [recheck_gradient.py.txt](/benchmark-data/energy-cache-optimization/recheck_gradient.py.txt)

<a href="/benchmark-data/energy-cache-optimization/raw-samples.tar.xz" download>All raw samples</a> (tar.xz; original JSONL files including numerical outputs, losslessly archived).

Process resource measurements:

- [0-baseline-composites.resources.txt](/benchmark-data/energy-cache-optimization/0-baseline-composites.resources.txt)
- [0-baseline-energy.resources.txt](/benchmark-data/energy-cache-optimization/0-baseline-energy.resources.txt)
- [0-baseline-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/0-baseline-equilibrium.resources.txt)
- [1-cache-composites.resources.txt](/benchmark-data/energy-cache-optimization/1-cache-composites.resources.txt)
- [1-cache-energy.resources.txt](/benchmark-data/energy-cache-optimization/1-cache-energy.resources.txt)
- [1-cache-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/1-cache-equilibrium.resources.txt)
- [2-cache-composites.resources.txt](/benchmark-data/energy-cache-optimization/2-cache-composites.resources.txt)
- [2-cache-energy.resources.txt](/benchmark-data/energy-cache-optimization/2-cache-energy.resources.txt)
- [2-cache-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/2-cache-equilibrium.resources.txt)
- [3-baseline-composites.resources.txt](/benchmark-data/energy-cache-optimization/3-baseline-composites.resources.txt)
- [3-baseline-energy.resources.txt](/benchmark-data/energy-cache-optimization/3-baseline-energy.resources.txt)
- [3-baseline-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/3-baseline-equilibrium.resources.txt)
- [exp-0-experiment2-cache-composites.resources.txt](/benchmark-data/energy-cache-optimization/exp-0-experiment2-cache-composites.resources.txt)
- [exp-0-experiment2-cache-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-0-experiment2-cache-equilibrium.resources.txt)
- [exp-0-experiment3-cache-assembly.resources.txt](/benchmark-data/energy-cache-optimization/exp-0-experiment3-cache-assembly.resources.txt)
- [exp-0-experiment3-cache-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-0-experiment3-cache-equilibrium.resources.txt)
- [exp-1-experiment2-experiment2-composites.resources.txt](/benchmark-data/energy-cache-optimization/exp-1-experiment2-experiment2-composites.resources.txt)
- [exp-1-experiment2-experiment2-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-1-experiment2-experiment2-equilibrium.resources.txt)
- [exp-1-experiment3-experiment3-assembly.resources.txt](/benchmark-data/energy-cache-optimization/exp-1-experiment3-experiment3-assembly.resources.txt)
- [exp-1-experiment3-experiment3-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-1-experiment3-experiment3-equilibrium.resources.txt)
- [exp-2-experiment2-experiment2-composites.resources.txt](/benchmark-data/energy-cache-optimization/exp-2-experiment2-experiment2-composites.resources.txt)
- [exp-2-experiment2-experiment2-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-2-experiment2-experiment2-equilibrium.resources.txt)
- [exp-2-experiment3-experiment3-assembly.resources.txt](/benchmark-data/energy-cache-optimization/exp-2-experiment3-experiment3-assembly.resources.txt)
- [exp-2-experiment3-experiment3-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-2-experiment3-experiment3-equilibrium.resources.txt)
- [exp-3-experiment2-cache-composites.resources.txt](/benchmark-data/energy-cache-optimization/exp-3-experiment2-cache-composites.resources.txt)
- [exp-3-experiment2-cache-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-3-experiment2-cache-equilibrium.resources.txt)
- [exp-3-experiment3-cache-assembly.resources.txt](/benchmark-data/energy-cache-optimization/exp-3-experiment3-cache-assembly.resources.txt)
- [exp-3-experiment3-cache-equilibrium.resources.txt](/benchmark-data/energy-cache-optimization/exp-3-experiment3-cache-equilibrium.resources.txt)
