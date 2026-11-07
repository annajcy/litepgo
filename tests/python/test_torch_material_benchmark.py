"""Correctness smoke checks for the benchmark, without timing thresholds."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

pytest.importorskip("resource", reason="benchmark RSS collection requires macOS or Linux")

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/torch_material.py"


def test_case_sweeps_are_unique_and_cover_scales():
    namespace = runpy.run_path(str(SCRIPT))
    full = namespace["cases"]("full")
    assert len(full) == len({json.dumps(case, sort_keys=True) for case in full})
    assert {case["size"] for case in full if case["layer"] == "fem"} >= {3, 6, 12}
    assert {case["weights"] for case in full if case["model"] == "neural"} == {2, 8, 32}


@pytest.mark.parametrize("layer,model,size,threads", [
    ("material", "native", 2, 1), ("material", "torch", 2, 1), ("material", "neural", 2, 1),
    ("fem", "torch", 1, 1), ("equilibrium", "neural", 1, 1), ("fem", "neural", 2, 4),
])
def test_worker_validates_and_records_samples(layer, model, size, threads):
    case = dict(layer=layer, model=model, size=size, batch=32, weights=2, threads=threads)
    run = subprocess.run([sys.executable, str(SCRIPT), "--case", json.dumps(case), "--repeats", "1"],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    records = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(records) == (2 if layer == "equilibrium" else 4)
    for record in records:
        assert record["status"] == "ok"
        assert record["median_ms"] == record["samples_ms"][0] > 0
        assert record["process_peak_mib"] >= record["import_peak_mib"] > 0


def test_timeout_preserves_a_record_for_each_case(tmp_path):
    run = subprocess.run([sys.executable, str(SCRIPT), "--timeout", "0.001", "--output", str(tmp_path)],
                         capture_output=True, text=True, timeout=30)
    records = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(records) == len(runpy.run_path(str(SCRIPT))["cases"]("pilot"))
    assert all(record["status"] == "timeout" for record in records)
    assert run.returncode == 2
    assert (tmp_path / "timings.csv").is_file()
