"""Layered CPU float64 benchmark. Each case runs in a fresh, timeout-limited process."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def cases(profile):
    """One-factor sweeps, not a Cartesian product of every expensive setting."""
    result = []
    def add(layer, size, model, batch=256, weights=2, threads=1):
        case = dict(layer=layer, size=size, model=model, batch=batch, weights=weights, threads=threads)
        if case not in result:
            result.append(case)
    for model in ("native", "torch", "neural"):
        for points in ((32,) if profile == "pilot" else (32, 256, 2048)):
            add("material", points, model)
        for subdivisions in ((1, 2) if profile == "pilot" else (3, 6, 12)):
            add("fem", subdivisions, model)
        for subdivisions in ((1,) if profile == "pilot" else (1, 2)):
            add("equilibrium", subdivisions, model)
        for batch in ((1, 32) if profile == "pilot" else (1, 32, 1024)):
            add("fem", 2 if profile == "pilot" else 6, model, batch=batch)
        add("fem", 2 if profile == "pilot" else 6, model, threads=4)
    for weights in (8, 32):
        add("material", 32 if profile == "pilot" else 256, "neural", weights=weights)
        add("fem", 1 if profile == "pilot" else 3, "neural", weights=weights)
        add("equilibrium", 1, "neural", weights=weights)
    return result


def peak_mib():
    divisor = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def measure(operation, repeats, prepare=lambda: None):
    samples = []
    for _ in range(repeats + 2):
        prepare()
        start = time.perf_counter()
        value = operation()
        samples.append(1000 * (time.perf_counter() - start))
        del value
    return dict(first_ms=samples[0], samples_ms=samples[2:],
                median_ms=statistics.median(samples[2:]), min_ms=min(samples[2:]), max_ms=max(samples[2:]))


def worker(case, repeats):
    import numpy as np
    import scipy.sparse as sp
    import torch
    from litepgo import build_info, dof, energy, fem, mesh, solver, tbb_thread_limit
    from litepgo.torch import response
    sys.path.insert(0, str(ROOT / "examples/fem/fitting"))
    from neural_equilibrium_fit import make_material
    from torch_neohookean import TorchNeoHookean
    from contact_scaling import generate

    torch.set_num_threads(case["threads"])
    torch.set_num_interop_threads(1)
    baseline = peak_mib()
    setup_start = time.perf_counter()
    diagnostic_ms = 0.
    k = case["weights"] if case["model"] == "neural" else 2
    if case["model"] == "neural":
        material, _, _, reference = make_material(k)
        p = reference.numpy().copy()
    else:
        material = fem.PointwiseVolumetricAdapter[fem.NeoHookean]() if case["model"] == "native" else TorchNeoHookean()
        p = np.array([2., 3.])
    metadata = dict(case, parameters=k, import_peak_mib=baseline, build=build_info(),
                    torch_version=torch.__version__, numpy_version=np.__version__)

    def native_density(F, Fp, theta, name):
        # Independent pointwise reference for validating batch outputs.
        method = getattr(fem.NeoHookean, "eval_" + name)
        return np.array([method(f, fp, t) for f, fp, t in zip(F, Fp, theta)])

    with tbb_thread_limit(case["threads"]):
        if case["layer"] == "material":
            n = case["size"]
            F = np.broadcast_to(np.array([[1.04, .02, 0], [0, .98, .01], [.01, 0, 1.03]]), (n, 3, 3)).copy()
            Fp = np.broadcast_to(np.eye(3), F.shape).copy()
            theta = np.tile(p, (n, 1))
            names = ("W", "dW_dF", "d2W_dF2", "d2W_dF_dtheta")
            operations = {}
            for name in names:
                operations[name] = lambda name=name: material.eval(F, Fp, theta, outputs=(name,))[name]
            metadata["points"] = n
            def validate(name, value):
                assert np.isfinite(value).all()
                if case["model"] == "torch":
                    expected = native_density(F, Fp, theta, name)
                    order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
                    if name == "d2W_dF2":
                        expected = expected[:, order][:, :, order].reshape(n, 3, 3, 3, 3)
                    elif name == "d2W_dF_dtheta":
                        expected = expected[:, order].reshape(n, 3, 3, k)
                    np.testing.assert_allclose(value, expected, rtol=1e-9, atol=1e-11)
                elif case["model"] == "neural":
                    one = material.eval(F[:1], Fp[:1], theta[:1], outputs=(name,))[name]
                    np.testing.assert_allclose(value, np.repeat(one, n, axis=0), rtol=1e-9, atol=1e-11)
            prepares = {}
        else:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "grid.veg"
                generate(case["size"], [(0., 1.)] * 3, path)
                volume = mesh.read_veg(path)
            X, topology = volume.positions, volume.topology
            mapping = dof.VertexDofMap(topology, 3)
            n = topology.num_cells
            geometry = [fem.InterpolatedResponse[fem.TetP1Basis](mapping.gather(e, X.ravel())) for e in range(n)]
            plastic = [energy.AffineResponse(np.zeros((9, 3)), np.eye(3).ravel())] * n
            key = (fem.TetP1Basis, fem.TetCentroidQuadrature)
            parameter_map = dof.ConstantDofMap(topology, k)
            u = (X @ np.array([[.04, .02, 0], [0, -.02, .01], [.01, 0, .03]]).T).ravel()
            def elastic(parameters, batch=case["batch"], kind=material):
                fields = [energy.AffineResponse(np.zeros((k, 3)), parameters)] * n
                return fem.VolumetricFemEnergy[key](kind, mapping, geometry, plastic, fields, batch)
            def forces(state, batch=case["batch"], kind=material):
                return fem.VolumetricFemForceMaterialResponse[key](kind, parameter_map, mapping, geometry, state, plastic, batch)
            obj = elastic(p)
            force = forces(u)
            upstream = np.linspace(-.2, .3, len(u))
            metadata.update(elements=n, dofs=len(u))
            prepares = {}
            if case["layer"] == "fem":
                operations = {"value": lambda: obj.eval_value(u), "gradient": lambda: obj.eval_gradient(u),
                              "hessian": lambda: obj.eval_hessian(u), "parameter_vjp": lambda: force.evaluate(p).vjp(upstream)}
                def validate(name, value):
                    array = value.data if sp.issparse(value) else value
                    assert np.isfinite(array).all()
                    # A different batch partition checks the whole assembled result.
                    other = elastic(p, batch=32 if case["batch"] != 32 else 256,
                                    kind=fem.PointwiseVolumetricAdapter[fem.NeoHookean]() if case["model"] == "torch" else material)
                    if name == "parameter_vjp":
                        expected = forces(u, batch=1, kind=fem.PointwiseVolumetricAdapter[fem.NeoHookean]() if case['model'] == 'torch' else material).evaluate(p).vjp(upstream)
                    else:
                        expected = getattr(other, "eval_" + name)(u)
                    if sp.issparse(value):
                        diff = value - expected
                        assert not diff.nnz or abs(diff).max() < 1e-9
                    else:
                        np.testing.assert_allclose(value, expected, rtol=1e-9, atol=1e-11)
            else:
                fixed_ids = np.array([3*i+j for i in np.flatnonzero(X[:, 0] == 0) for j in range(3)])
                fixed = dof.DofReduction(len(u), fixed_ids)
                free = np.setdiff1d(np.arange(len(u)), fixed_ids)
                load = np.zeros_like(X)
                loaded = np.flatnonzero(X[:, 0] == 1)
                load[loaded] = np.array([.015, -.01, .02]) / len(loaded)
                external = energy.QuadraticEnergy(sp.csc_matrix((len(u), len(u))), -load.ravel())
                def total(parameters):
                    return energy.WeightedSumEnergy([(elastic(parameters), 1.), (external, 1.)])
                options = solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=50)
                initial = np.zeros_like(u)
                eq = solver.StaticEquilibriumResponse(k, fixed, initial, options, total, forces)
                # Separate diagnostic solve; the timed bridge uses the same settings.
                diagnostic_start = time.perf_counter()
                diagnostic = solver.newton_solve(total(p), fixed, initial, options)
                if diagnostic.status != solver.NewtonStatus.Converged:
                    raise ValueError(f"equilibrium failed: {diagnostic.status}")
                metadata.update(newton_iterations=diagnostic.iterations, convergence=str(diagnostic.status),
                                free_residual=float(np.linalg.norm(total(p).eval_gradient(diagnostic.x)[free])))
                diagnostic_ms = 1000 * (time.perf_counter() - diagnostic_start)
                metadata["diagnostic_ms"] = diagnostic_ms
                tensor = torch.tensor(p, requires_grad=True)
                saved_loss = None
                def prepare_backward():
                    nonlocal saved_loss
                    tensor.grad = None
                    state = response(eq, tensor)
                    saved_loss = .5 * state[free].square().mean()
                def backward():
                    saved_loss.backward()
                    return tensor.grad
                operations = {"forward": lambda: eq.evaluate(p).value, "loss_backward": backward}
                prepares["loss_backward"] = prepare_backward
                def validate(name, value):
                    if name == "forward":
                        np.testing.assert_allclose(value, diagnostic.x, atol=1e-10)
                    else:
                        assert torch.isfinite(value).all()
                        direction = np.linspace(.3, .7, k)
                        def loss(q):
                            return .5 * np.mean(eq.evaluate(q).value[free] ** 2)
                        fd = (loss(p + 1e-4*direction) - loss(p - 1e-4*direction)) / 2e-4
                        np.testing.assert_allclose(value.detach().numpy() @ direction, fd, rtol=2e-4, atol=1e-9)
        metadata["setup_ms"] = 1000 * (time.perf_counter() - setup_start) - diagnostic_ms
        for name, operation in operations.items():
            prepare = prepares.get(name, lambda: None)
            # First measured call precedes validation; steady samples follow validation.
            prepare()
            start = time.perf_counter()
            value = operation()
            cold = 1000 * (time.perf_counter() - start)
            validate(name, value)
            del value
            measured = measure(operation, repeats, prepare)
            measured["first_ms"] = cold
            print(json.dumps(dict(metadata, operation=name, status="ok", **measured,
                                  process_peak_mib=peak_mib())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--output", type=Path, default=ROOT / "output/torch_material_benchmark")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.timeout < float("inf"):
        parser.error("repeats and timeout must be positive and timeout finite")
    if args.case:
        worker(json.loads(args.case), args.repeats)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    metadata = dict(platform=platform.platform(), machine=platform.machine(), cpu_count=os.cpu_count(),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    python=sys.version, executable=sys.executable, profile=args.profile, repeats=args.repeats,
                    timeout=args.timeout, command=sys.argv, thread_environment={k: env[k] for k in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    metadata["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    metadata["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__).resolve(), ROOT / "benchmarks/contact_scaling.py",
                                             ROOT / "src/python/torch.py", ROOT / "examples/fem/fitting/neural_neohookean.py",
                                             ROOT / "examples/fem/fitting/neural_equilibrium_fit.py",
                                             ROOT / "examples/fem/fitting/torch_neohookean.py")}
    if sys.platform == "darwin":
        metadata["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    with (args.output / "results.jsonl").open("w") as stream:
        for case in cases(args.profile):
            print(case, flush=True)
            command = [sys.executable, str(Path(__file__).resolve()), "--case", json.dumps(case), "--repeats", str(args.repeats)]
            try:
                run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=args.timeout)
                output, failure = run.stdout, run.stderr if run.returncode else None
                status = "error" if failure is not None else "ok"
            except subprocess.TimeoutExpired as error:
                output = error.stdout or b""
                if isinstance(output, bytes):
                    output = output.decode()
                failure, status = f"exceeded {args.timeout}s per case", "timeout"
            for line in output.splitlines():
                record = json.loads(line)
                records.append(record)
                stream.write(json.dumps(record) + "\n")
            if status != "ok":
                record = dict(case, status=status, error=failure)
                records.append(record)
                stream.write(json.dumps(record) + "\n")
            stream.flush()
    fields = ["layer", "size", "model", "batch", "weights", "threads", "operation", "status", "elements",
              "setup_ms", "first_ms", "median_ms", "min_ms", "max_ms", "import_peak_mib", "process_peak_mib", "newton_iterations"]
    with (args.output / "timings.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} records to {args.output}")
    if any(r["status"] == "error" for r in records):
        raise SystemExit(1)
    if any(r["status"] == "timeout" for r in records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
