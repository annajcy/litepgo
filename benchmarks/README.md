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
