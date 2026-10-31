# Open: Python mesh-energy callback overhead

Status: **partially mitigated; open**. Recorded 2026-10-31.

Python `MeshEnergy` invokes Python once per stencil. For inexpensive edge terms,
callback dispatch, GIL acquisition and array allocation/conversion dominate the
local arithmetic. `BatchedMeshEnergy` substantially reduces this overhead, but
still trails native C++ `MeshEnergy`. This is a performance limitation, not a
known numerical discrepancy.

## Measurements

Apple M1 Pro (8 CPU cores), macOS arm64, Release (`-O3 -DNDEBUG`), default TBB
thread configuration. Uniform triangle-grid edge energy is
`0.5 * sum_edges ||x_i-x_j||^2`, each undirected edge counted once. Native and
Python results were checked numerically. These are wall-clock observations on
one machine, not a cross-platform guarantee.

4,225 vertices / 12,416 edges; milliseconds per evaluation:

| Batch size | Value | Gradient | Exact Hessian |
| --- | ---: | ---: | ---: |
| 32 | 3.183 | 3.835 | 5.338 |
| 64 | 2.182 | 2.663 | 3.968 |
| 128 | 1.744 | 2.070 | 3.218 |
| 256 | 1.527 | 1.817 | 2.763 |
| 512 | 1.379 | 1.650 | 2.488 |
| 1024 | 1.309 | 1.565 | 2.389 |
| 2048 | 1.295 | 1.565 | 2.342 |
| All 12,416 | 1.241 | 1.484 | 2.317 |
| Native C++ (per-term callback) | 0.202 | 0.173 | 0.915 |

At the best measured batch size, Python takes approximately 6.2x / 8.6x / 2.5x
native time. Improvement beyond 1024–2048 is modest. The default remains 256:
this edge-only measurement does not establish an optimal default for variable
one-rings or larger FEM terms.

A separate per-term Python run measured 118.6 / 139.1 / 140.3 ms. This establishes
why batching matters; do not combine timings from separate runs as if their
native baselines were identical. Raw records:

- [Batch-size sweep](batch-size-sweep.json), including 81- and 1,089-vertex grids.
- [Per-term Python callbacks](scalar-callback.json).
- [Per-term ARAP samples](arap-scalar-seconds.json).

The 81-vertex ARAP example converges in 373 iterations. Whole-process medians
include startup, rotations, energy checks, solves and OBJ output:

| Path | Seconds |
| --- | ---: |
| Native C++ | 0.235 |
| Python batch 32 | 1.345 |
| Python batch 64 | 1.283 |
| Python batch 128 | 1.219 |
| Python batch >=256 | 1.200–1.215 |
| Python per-term (separate run) | 5.078 |

ARAP has only 208 edges: batch sizes >=256 all invoke one callback per evaluation.
Differences among those settings are timing variation, not batch-size benefits.
The remaining approximately 5.1x whole-program gap includes Python algorithm
loops; it cannot be attributed entirely to binding overhead.

## Method and limits

The batch sweep warms each evaluation twice, shuffles batch-size order with seed
0, and measures nine rounds of five evaluations, reporting the median per-call
round average. Native evaluation uses two warmups and seven rounds of ten calls.
The native gradient/Hessian benchmark also computes an output norm inside the
measurement; Python computes this check outside timing, slightly favoring Python.
ARAP warms once and measures three runs per setting in shuffled order.
Compilation and energy construction are excluded from evaluation timings;
whole-program ARAP includes construction. Results are not a statistical claim
that tiny timing differences are significant.

No batch-size sweep has yet been run for variable-length one-rings, nonlinear
FEM terms, explicit single-thread execution, or native C++ batched callbacks.
Peak memory and allocation counts have not been measured.

## Remaining work

1. Profile gathering, conversion, callback execution, projection and assembly
   separately before attributing the remaining gap.
2. Remove the extra gather/copy pass in `BatchedMeshEnergy` if profiling supports
   it. Currently batch outputs are cached per term and passed through an internal
   `MeshEnergy` to reuse validation, per-term Hessian projection and assembly.
   This also retains duplicate derivative storage. Batch size limits temporary
   batch storage, not total assembly memory.
3. Investigate fewer array conversions and buffer allocations without weakening
   the promise that Python may retain input arrays safely.
4. Measure one-ring and larger terms before changing the default batch size.
5. Profile ARAP's Python rotation/objective loops independently. Constant-Hessian
   and factorization reuse are separate solver optimizations, not binding fixes.

Any optimization must preserve original term IDs/order, variable stencil sizes,
vertex-major derivative layout, independent gradient/Hessian requests, per-term
PSD projection, fixed handles, callback exceptions and input lifetimes. Repeat
numerical equivalence and the same benchmark workloads after changes.
