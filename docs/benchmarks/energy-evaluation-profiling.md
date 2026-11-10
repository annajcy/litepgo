# Energy evaluation: profiling and corrected baseline

The Debug audit found two defects: matrix coefficients were inspected before
checking compressed structure, causing an Eigen assertion instead of an exception;
and returning a structured-binding Jacobian by name copied it, incidentally
compressing invalid output before validation. Both are fixed. A new native test
covers uncompressed exact and projected Energy results. Debug Energy/Response
86/86 and solver 35/35 pass; Release CTest 507/507 passes (Python 658 passed,
81 skipped). These fixes are separate from performance optimization.

## What explains the overhead

A fresh evaluation allocates state, snapshots the input, and retains cached
results. A two-term WeightedSumEnergy creates three evaluations and three input
snapshots. Gradient/Hessian methods return owning values, so internal consumers
also copy cached derivatives. The old joint result moved outputs rather than
retaining and copying them. Python joint evaluation now crosses the binding for
`evaluate`, `gradient` and `value` instead of one `eval` call.

These are identifiable added operations, not measured percentages of total cost.
In particular, cached-gradient timing includes Python array conversion; it is
not a pure C++ copy measurement. There is no evidence of repeated constitutive
assembly caused by reading the cached value, or repeated equilibrium solves.

The original small-shell case is a weighted sum of shell elasticity and a
quadratic foundation, not just a ShellFemEnergy. The following decomposition uses
8 shell faces / 27 DOFs and an otherwise equivalent zero-linear-term foundation.
Before/after/after/before runs use 200 warmups and 31 samples of 200 calls per
process. Times are pooled medians in microseconds:

| Threads | Model | Old joint call | New fresh evaluation | Difference | Create/destroy state only |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | leaf | 8.217 | 9.088 | +0.871 | 0.404 |
| 1 | sum_one | 8.398 | 9.368 | +0.969 | 0.508 |
| 1 | sum_two | 8.570 | 9.683 | +1.113 | 0.616 |
| 1 | quadratic | 0.993 | 1.288 | +0.296 | 0.396 |
| 4 | leaf | 13.438 | 14.485 | +1.047 | 0.399 |
| 4 | sum_one | 14.199 | 13.999 | -0.200 | 0.506 |
| 4 | sum_two | 14.510 | 14.967 | +0.457 | 0.618 |
| 4 | quadratic | 0.976 | 1.337 | +0.361 | 0.401 |

`leaf` is the native shell, `sum_one` wraps it in one weighted term, and `sum_two`
adds the foundation. The single-thread cases show roughly 0.9–1.1 μs added to
shell/composite calls. Creating the composite state alone costs about 0.62 μs.
These timings do not add exactly: the old path also allocated outputs, and the
isolated phases have different object lifetimes and cache behavior.

## Why the four-thread percentages vary

Both binaries spend substantial sampled main-thread time inside the mesh batch
TBB execute-and-wait subtree. That subtree includes useful work, so its inclusive
sample count must not be called pure waiting time. The tiny workload uses small
batch groups plus parallel validation of only eight element views; assembly of
27 DOFs fits in one 64-column grain. Four threads already cost more than one in
the old implementation. Dispatch, work stealing, timing cadence and scheduler
state therefore matter greatly relative to the few microseconds of work.

The earlier +30.5% observation was real in those runs but is **not a stable
migration penalty**. Keeping the original problem and foundation load exactly,
a longer recheck (200 warmups, 31 samples × 200 calls; before/after/after/before)
gives:

| Original case | Before μs | Corrected current μs | Change |
| --- | ---: | ---: | ---: |
| tet_torch | 223.678 | 222.439 | -0.6% |
| shell_gradient | 20.714 | 18.889 | -8.8% |

The sign changes across experiments. This does not establish a speedup either;
it shows why the earlier percentage should not be treated as a fixed cost of
the API. The original benchmark and all rechecks remain available.

## Next optimization targets

1. Avoid copying cached vectors/matrices between native composite/equilibrium
   internals; keep Python outputs independently owned.
2. Reduce state/control-block allocation and redundant input snapshots without
   changing the independent-evaluation lifetime contract.
3. Test a serial path for very small batch/assembly validation workloads. This
   targets a pre-existing scheduling cost, not a newly changed material formula.

No such optimization has been included in the corrected baseline. These are
priorities supported by the call paths, not promised speedups. Native phase-level
allocation accounting and a serial-path ablation would be needed to attribute
all of the observed timing difference precisely.

## Corrected baseline

After fixing the two correctness issues and finishing all tests, two separate
process runs collected 48 Energy cases and 98 equilibrium/Response cases each.
Energy cases pool 30 samples, equilibrium 14, and tiny Response controls 40.
All values, gradients, Hessian actions and equilibrium states agree with the
previous migration results (`rtol=2e-8`, `atol=1e-9`). Each equilibrium operation
still performs one nonlinear solve. No builds, tests or sampling ran alongside
these baseline timings. The isolated local module is
`build/energy-profile/fixed-python`; its wrappers are copied, not symlinked.

- [Corrected baseline CSV](/benchmark-data/energy-evaluation-fixed/baseline.csv)
- [Binary/source fingerprints and validation](/benchmark-data/energy-evaluation-fixed/environment.json)
- [Decomposition script](/benchmark-data/energy-evaluation-fixed/phases.py.txt)
- [Exact-problem longer recheck](/benchmark-data/energy-evaluation-fixed/recheck_long.py.txt)
- [Sampling workload](/benchmark-data/energy-evaluation-fixed/hot.py.txt)
- [Old process sample](/benchmark-data/energy-evaluation-fixed/sample-before.txt)
- [Migration process sample](/benchmark-data/energy-evaluation-fixed/sample-after.txt)

Raw baseline runs:

- [0: energy samples](/benchmark-data/energy-evaluation-fixed/fixed-0-energy.jsonl.txt), [resources](/benchmark-data/energy-evaluation-fixed/fixed-0-energy.resources.txt)
- [0: equilibrium samples](/benchmark-data/energy-evaluation-fixed/fixed-0-equilibrium.jsonl.txt), [resources](/benchmark-data/energy-evaluation-fixed/fixed-0-equilibrium.resources.txt)
- [1: energy samples](/benchmark-data/energy-evaluation-fixed/fixed-1-energy.jsonl.txt), [resources](/benchmark-data/energy-evaluation-fixed/fixed-1-energy.resources.txt)
- [1: equilibrium samples](/benchmark-data/energy-evaluation-fixed/fixed-1-equilibrium.jsonl.txt), [resources](/benchmark-data/energy-evaluation-fixed/fixed-1-equilibrium.resources.txt)

Raw decomposition and longer rechecks:

- [0: decomposition](/benchmark-data/energy-evaluation-fixed/phases-0-before.jsonl.txt), [long recheck](/benchmark-data/energy-evaluation-fixed/long-0-before.jsonl.txt)
- [1: decomposition](/benchmark-data/energy-evaluation-fixed/phases-1-after.jsonl.txt), [long recheck](/benchmark-data/energy-evaluation-fixed/long-1-after.jsonl.txt)
- [2: decomposition](/benchmark-data/energy-evaluation-fixed/phases-2-after.jsonl.txt), [long recheck](/benchmark-data/energy-evaluation-fixed/long-2-after.jsonl.txt)
- [3: decomposition](/benchmark-data/energy-evaluation-fixed/phases-3-before.jsonl.txt), [long recheck](/benchmark-data/energy-evaluation-fixed/long-3-before.jsonl.txt)

Current drivers: [Energy](/benchmark-data/energy-evaluation-fixed/energy_evaluation.py.txt), [equilibrium](/benchmark-data/energy-evaluation-fixed/equilibrium_evaluation.py.txt).
