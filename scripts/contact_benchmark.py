#!/usr/bin/env python3
"""Run: python3 scripts/contact_benchmark.py [build-directory]

Builds Release with both backends, then runs the exact tet_p1_box_stack setup.
Outputs full precision CSVs, displacement arrays, traces and summary.json under
output/contact_benchmark by default. Round 0 is warmup; five measured rounds
by default, alternating backend order. See --help for mesh/thread/output options. Timings
exclude mesh I/O, construction and diagnostics, include Newton symbolic analysis
and a preallocated in-memory iteration trace. Shared-state derivatives are of
kappa * contact energy in surface position coordinates; Hessian RMS includes zeros.
shared_ccd.csv uses one common direction, twice the native final displacement.
Existing output files with the same names are replaced on each invocation.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build", nargs="?", default=str(ROOT / "build/release"))
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--repetitions", type=int, default=5)
parser.add_argument("--mesh", type=Path, default=ROOT / "assets/box-tet.veg")
parser.add_argument("--output", type=Path, default=ROOT / "output/contact_benchmark")
parser.add_argument("--total-stiffness", type=float, default=0)
parser.add_argument("--skip-build", action="store_true")
parser.add_argument("--comparison", choices=("ipctk", "spatial-hash"), default="ipctk",
                    help="Compare native SAP with IPCTK (default) or native SpatialHash")
args = parser.parse_args()
other_backend = "spatial_hash" if args.comparison == "spatial-hash" else "ipctk"
if args.threads < 1 or args.repetitions < 1 or not math.isfinite(args.total_stiffness) or args.total_stiffness < 0:
    parser.error("threads/repetitions must be positive and total stiffness finite/nonnegative")
BUILD = Path(args.build).resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
ENV = dict(os.environ, VECLIB_MAXIMUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")


def command(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def rows(name):
    with (OUT / name).open() as stream:
        return list(csv.DictReader(stream))


if not args.skip_build:
    with (OUT / "build.log").open("w") as log:
        subprocess.run(["cmake", "-S", str(ROOT), "-B", str(BUILD),
                        "-DCMAKE_BUILD_TYPE=Release", "-DLITEPGO_WITH_IPC_TOOLKIT=ON"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(["cmake", "--build", str(BUILD), "--target", "contact_benchmark", "-j", "4"],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
mesh = args.mesh.resolve()
executable = BUILD / "example/contact/contact_benchmark"
metadata = {
    "platform": platform.platform(), "machine": platform.machine(),
    "cpu_count": os.cpu_count(), "git_head": command(["git", "rev-parse", "HEAD"]),
    "git_status": command(["git", "status", "--short"]),
    "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [ROOT / "example/contact/tet_p1_box_stack.cpp",
                                   ROOT / "example/contact/contact_benchmark.hpp",
                                   ROOT / "cpp/litepgo/contact/ipc_contact_energy.cpp",
                                   ROOT / "cpp/litepgo/contact/ipctk_contact_energy.cpp",
                                   ROOT / "cpp/litepgo/contact/broad_phase/sweep_and_prune.cpp",
                                   ROOT / "cpp/litepgo/contact/broad_phase/spatial_hash.cpp"]},
    "thread_environment": {key: ENV[key] for key in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    "build_directory": str(BUILD),
    "comparison": args.comparison,
    "threads": args.threads, "repetitions": args.repetitions, "total_stiffness": args.total_stiffness,
    "protocol": "one warmup per backend, then measured pairs with alternating backend order; steady_clock around newton_solve only",
}
if sys.platform == "darwin":
    metadata["cpu"] = command(["sysctl", "-n", "machdep.cpu.brand_string"])
    metadata["memory_bytes"] = int(command(["sysctl", "-n", "hw.memsize"]))
    (OUT / "processes_before.txt").write_text(command(["ps", "-Ao", "pid,pcpu,comm"]) + "\n")
(OUT / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
(OUT / "CMakeCache.txt").write_bytes((BUILD / "CMakeCache.txt").read_bytes())
with (OUT / "run.log").open("w") as log:
    completed = subprocess.run([str(executable), str(mesh), str(OUT), str(args.threads), str(args.repetitions), str(args.total_stiffness), args.comparison],
                               cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
results = rows("runs.csv")
assert len(results) == 2 * (args.repetitions + 1), "Expected one warmup and requested measured runs per backend"
summary = {}
for backend in ("native", other_backend):
    measured = [r for r in results if r["backend"] == backend and int(r["round"]) > 0]
    assert len(measured) == args.repetitions
    assert all(int(r["status"]) == 0 and float(r["free_residual"]) <= 1e-8 for r in measured), "Not converged"
    times = [float(r["seconds"]) for r in measured]
    summary[backend] = {
        "seconds_median": statistics.median(times), "seconds_mean": statistics.mean(times),
        "seconds_stdev": statistics.stdev(times) if len(times) > 1 else 0, "seconds_min": min(times), "seconds_max": max(times),
        "max_repeat_displacement": max(float(r["repeat_max_abs"]) for r in measured),
        "first_measured": measured[0],
    }
# Independent check of the C++ array comparison and saved full precision results.
x = [[float(r["displacement"]) for r in rows(f"{name}_0_displacement.csv")] for name in ("native", other_backend)]
assert len(x[0]) == len(x[1]) > 0 and all(math.isfinite(v) for a in x for v in a)
difference = [a - b for a, b in zip(*x)]
comparison = rows("displacement_comparison.csv")[0]
assert math.isclose(max(abs(v) for v in difference), float(comparison["max_abs"]), rel_tol=1e-12, abs_tol=1e-30)
assert math.isclose(math.sqrt(sum(v*v for v in difference) / len(difference)), float(comparison["rms"]), rel_tol=1e-12, abs_tol=1e-30)
if args.comparison == "spatial-hash":
    # Compare every saved field, not merely the reconstructed max/RMS differences.
    for round_index in range(args.repetitions + 1):
        pair = [next(r for r in results if r["backend"] == backend and int(r["round"]) == round_index)
                for backend in ("native", other_backend)]
        assert {k: v for k, v in pair[0].items() if k not in ("backend", "seconds")} == {
            k: v for k, v in pair[1].items() if k not in ("backend", "seconds")}, ("run mismatch", round_index)
        assert (OUT / f"native_{round_index}_displacement.csv").read_bytes() == (
            OUT / f"{other_backend}_{round_index}_displacement.csv").read_bytes(), ("displacement mismatch", round_index)
    traces = rows("iterations.csv")
    paths = [[{k: v for k, v in r.items() if k != "backend"} for r in traces if r["backend"] == backend]
             for backend in ("native", other_backend)]
    assert paths[0] == paths[1], "SAP/hash iteration trace mismatch"
    summary["sap_hash_all_non_timing_exact"] = True
summary["displacement_comparison"] = comparison
summary[f"native_over_{other_backend}_median_time"] = summary["native"]["seconds_median"] / summary[other_backend]["seconds_median"]
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
completed.check_returncode()
