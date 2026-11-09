# Response evaluation state performance

The Response API now binds values and derivatives to `response.evaluate(x)`.
Static equilibrium states retain the converged displacement and lazily prepare
exact stiffness, its factorization and the parameter-force response. Torch and
composed responses retain this state through backward. Explicit state-Jacobian
storage is allocated only when requested.

## Conditions and checks

Baseline: `0fae743`, with the pre-migration Python module preserved separately.
Both builds use Release `-O3 -DNDEBUG`, the same compiler, Accelerate and Eigen
vectorization. The same Python environment and mesh/problem construction are used.
Runs alternate before/after/after/before, with seven samples per case per run
(14 per version), after warmup. No builds, tests or tutorial executions overlap
these runs. Each sample creates a fresh computation graph and starts equilibrium
from the configured initial displacement. There is no cross-sample state reuse.

Cases cover TetP1, HexHermite, shell theta and shell Fp; 1/2 subdivisions per axis;
1/4 TBB threads; Exact/Projected Newton Hessians; standalone VJP, Torch
forward/backward and composed forward/backward. This gives 96 cases. Sizes are
6/48 tetrahedra, 1/8 hexahedra and 2/8 shell faces. The x=0 boundary is fixed;
small tip loads and an elastic foundation stabilize shell and underintegrated
Hermite modes. These are small controlled workloads, not large production meshes.

Every case checks free residual <= 1.1e-9. Small cases also check JVP against
central differences. Across all four runs, converged states and gradients match
the baseline within the tolerances enforced by the comparison script. Separate
regression tests cover two forwards with reverse-order backward, retained-owner
lifetime, copied inputs/outputs, exact versus projected Hessian reuse, and
invalid input/derivative results.

## Results

Median percentage change across sizes, thread counts and Hessian kinds; negative
means faster. Per-case milliseconds and raw samples are linked below.

| Model | Standalone VJP | Torch forward/backward | Composed forward/backward |
| --- | ---: | ---: | ---: |
| tet | +0.3% | -40.0% | -56.9% |
| hermite | -2.9% | -48.5% | -63.8% |
| shell | -8.7% | -43.8% | -57.2% |
| shell_fp | -2.3% | -40.7% | -58.8% |

Across individual cases, standalone VJP ranges from 18.8% faster to 6.2% slower;
Torch forward/backward is 32.4–50.6% faster and composed forward/backward is
40.7–64.5% faster. Small differences can include timing noise.

Solve counts are verified for every measured sample: standalone VJP remains
one nonlinear solve; Torch forward/backward drops from two to one; composed
forward/backward drops from three to one. Exact Newton reuses the converged
Hessian; Projected Newton still evaluates the exact Hessian for sensitivity.
The tests verify Hessian request counts. This benchmark does not separately time
assembly, symbolic analysis, factorization and triangular solves.

## Small-response overhead and memory

Three-component affine value calls: 0.801 → 1.069 microseconds (+33.4%). Three-component exp value calls: 0.748 → 1.057 microseconds (+41.4%). The new one-shot call creates an evaluation, copies its input and returns an
owned Python value. This is a measured cost of the unified API; small pure-value
calls did not improve. Retaining an evaluation avoids recomputing its cached
value but also keeps its model/state alive.

Whole-process peak RSS (including Python/Torch startup) was 288.8/291.4 MiB before and 292.7/299.2 MiB after. These process peaks do not isolate retained-factorization memory. Large graphs
holding many equilibrium evaluations can use more memory; release states/graphs
when finished. The measurements do not establish large-mesh scaling or general
speedup bounds.

## Reproduce

Run `python benchmarks/equilibrium_evaluation.py --samples 7` with the selected
build on `PYTHONPATH`. The archived baseline script uses the old API and should
only be run with the baseline build. Raw samples, gradients, states, solve counts,
residuals and process resource measurements are in
[benchmark data](/benchmark-data/response-evaluation/comparison.csv).

- [Environment](/benchmark-data/response-evaluation/environment.json)
- [Baseline script](/benchmark-data/response-evaluation/baseline-script.py.txt)
- [Before run 1](/benchmark-data/response-evaluation/paired-0-before.jsonl.txt)
- [After run 1](/benchmark-data/response-evaluation/paired-1-after.jsonl.txt)
- [After run 2](/benchmark-data/response-evaluation/paired-2-after.jsonl.txt)
- [Before run 2](/benchmark-data/response-evaluation/paired-3-before.jsonl.txt)

Validation: 506/506 CTest entries, including Python (646 passed, 81 skipped,
one existing Torch deprecation warning); all 11 tutorials executed, including
the two C++ notebooks; reference generation and VitePress build passed.
