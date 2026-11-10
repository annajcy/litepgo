# Benchmarks

Run from the repository root with the release build selected. The full protocols
and recorded results are in [the benchmark documentation](../docs/benchmarks/index.md).

| Driver | Purpose |
| --- | --- |
| `torch_material.py` | CPU float64 material, assembled FEM, and equilibrium backward; isolated cases, correctness checks, timings, and peak RSS |
| `shell_batch_benchmark.py` | Native shell value/gradient/Hessian batch/thread sweep with numerical consistency checks |
| `mesh_batch_benchmark.py` | Native FEM batch/thread sweep; CSV on stdout |
| `contact_benchmark.py` | Native/IPCTK contact comparison; builds the C++ driver unless `--skip-build` |
| `contact_scaling.py` | Validated structured mesh generation and contact mesh/thread scaling |

```sh
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/torch_material.py
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/torch_material.py --profile full --timeout 90 --output output/torch_material_full
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/mesh_batch_benchmark.py
uv run --no-sync python benchmarks/contact_benchmark.py --help
uv run --no-sync python benchmarks/contact_scaling.py --help
```

The three older Python drivers moved from `scripts/`; update external invocations.
The shared C++ contact driver remains in `examples/contact/` so its geometry and
loading cannot diverge from the example. Its CMake target and executable path did
not change. Output directories remain under `output/`, ignored by Git. Each run
replaces files in the selected output directory; choose a new directory to retain
an earlier run. The Torch suite's peak-RSS collector requires macOS or Linux.

Native force-product benchmark: `python benchmarks/force_products.py` measures
JVP/VJP for HexQ1 and HexHermite material/Fp forces, shell material/Fp forces,
and spatial deformation-gradient queries. It uses one TBB thread, batches 1/256,
30 samples of 10 calls, and checks products against explicit Jacobians before
timing. Run before and after with the same Release build settings, without
concurrent builds or tests. Output is JSON Lines with all raw samples.

Response evaluation / equilibrium benchmark: `python benchmarks/equilibrium_evaluation.py --samples 7`.
It covers TetP1, HexHermite, shell material and shell Fp with fixed 1/4-thread
runs, Exact/Projected Newton, standalone VJP and Torch/composed forward/backward.
Each timing starts with a fresh evaluation/graph. It checks residuals and small-case
finite differences, records solve counts and gradients, and also measures tiny
pure-value responses. See `docs/benchmarks/response-evaluation.md` for the paired
baseline comparison and its small-mesh limitations.

Energy evaluation benchmark: `python benchmarks/energy_evaluation.py --samples 15`.
It covers tiny and 1000-DOF quadratic energies plus TetP1, HexHermite and shell FEM
with 1/4 TBB threads. Value, value/gradient, joint exact/projected derivatives,
and Hessian-only calls are measured separately. Each timing creates a fresh
evaluation; repeated access inside that evaluation uses its cache. Hessian-only
requests also prepare gradient/value under the new contract. The comparison uses
an archived old-API driver and checks values, gradients, Hessian actions and
end-to-end equilibrium results. See `docs/benchmarks/energy-evaluation.md` for
raw samples, build conditions and measured overheads. Do not run concurrently
with builds, tests, or tutorial execution.

Focused cache/assembly checks: `python benchmarks/energy_cache.py --suite composites`
measures fresh-state construction, gradients and Hessians for leaf, weighted,
nested and affine-mapped energies at 3/1000/10000 DOFs. `--suite assembly` measures
small shell, TetP1 and HexHermite gradients/Hessians with 1/4 TBB threads. Both
use warmups, repeated calls and numerical checks outside the timed region;
`PYTHONPATH` selects an isolated build for paired comparisons. See
`docs/benchmarks/energy-cache-optimization.md` for the cache optimization and
separate allocation/serial-path experiments.
