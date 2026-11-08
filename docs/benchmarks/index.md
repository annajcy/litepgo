# Benchmarks

Benchmark drivers live in `benchmarks/`. They measure specific workloads after
checking numerical results; their timing values are not CI thresholds. Use a
[release build](../guide/build.md), record the hardware and thread configuration,
and avoid concurrent test/build jobs when collecting a baseline.

## Suites

| Suite | Measures | Output |
| --- | --- | --- |
| [Torch material performance](./torch-material.md) | Constitutive evaluation, assembled FEM, parameter VJP, equilibrium forward and backward | Raw JSONL samples, CSV summaries, environment metadata |
| Native mesh batching | Value, gradient, Hessian over batch sizes and TBB thread counts | CSV on stdout |
| Contact backend comparison | Native and IPC Toolkit contact on the same example setup | CSV, displacement arrays, convergence traces, environment metadata |
| Contact scaling | Mesh-size/thread matrix using validated structured meshes | Per-case outputs and reports |

## Torch materials

Start with the small pilot. Use the full profile for mesh, material-point, batch,
and active-weight scaling; slow cases are stopped independently and reported.

```sh
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/torch_material.py
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/torch_material.py --profile full --timeout 90 --output output/torch_material_full
```

See [the protocol and measured results](./torch-material.md). The RSS collector
uses `resource.getrusage` on macOS/Linux; this suite is not a Windows RSS benchmark.

## Native mesh batching

```sh
mkdir -p output
PYTHONPATH=build/release/python uv run --no-sync python benchmarks/mesh_batch_benchmark.py > output/mesh_batch.csv
```

This existing microbenchmark uses 128 and 4,096 **disconnected tetrahedra**, batches
of 1/16/64/256 elements, and 1/2/4/8 TBB threads. It checks gradient invariance under
batch changes, warms up each operation, then reports the median of nine calls.
It is an assembly microbenchmark, not a connected-mesh equilibrium scaling study.

## Contact backends and scaling

```sh
uv run --no-sync python benchmarks/contact_benchmark.py --help
uv run --no-sync python benchmarks/contact_benchmark.py --threads 1 --repetitions 5 --output output/contact_benchmark
uv run --no-sync python benchmarks/contact_scaling.py --help
uv run --no-sync python benchmarks/contact_scaling.py --output output/contact_scaling
```

The comparison driver configures Release with IPC Toolkit enabled and builds
`contact_benchmark`, unless `--skip-build` is passed. It runs the same
`tet_p1_box_stack.cpp` scene for both backends, alternating their order after warmup.
It records convergence, derivative agreement, hashes and thread settings. The
scaling driver uses the asset mesh plus generated structured meshes, and 1/4/8
threads. These longer contact runs are separate from the Torch suite.

## Migration and reproducibility

`mesh_batch_benchmark.py`, `contact_benchmark.py`, and `contact_scaling.py` moved
from `scripts/` to `benchmarks/`. Their internal paths are updated; external scripts
using the old paths must be updated. There are no compatibility wrappers.
The contact C++ driver/header remain beside their shared scene in `examples/contact/`;
the target name, executable location and default output locations are unchanged.

Output directories are ignored by Git and reused files are replaced. Select a new
`--output` directory to preserve earlier measurements. Historical contact output
was not moved or rerun as part of this reorganization.
