"""Run the Python-material examples and check the agreement they report."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def run(relative, *args):
    path = ROOT / "examples" / relative
    completed = subprocess.run(
        [sys.executable, str(path), *args], capture_output=True, text=True, timeout=300, cwd=path.parent
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def errors(output):
    """Every reported *_error value."""
    return {name: float(value) for name, value in re.findall(r"(\w*error)=([-+0-9.e]+)", output)}


@pytest.mark.parametrize(
    "example,expected,tolerance",
    [
        ("fem/material/numpy_neohookean.py", 8, 1e-9),
        ("fem/material/numpy_koiter_shell.py", 8, 1e-12),
        ("fem/fitting/torch_neohookean.py", 13, 1e-12),
        ("fem/fitting/torch_equilibrium_response.py", 8, 1e-12),
    ],
)
def test_example_matches_native(example, expected, tolerance):
    # Keep the Hermite solve practical in unoptimized Debug builds.
    args = (ROOT / "tests/assets/box-hex.veg",) if example == "fem/material/numpy_neohookean.py" else ()
    output = run(example, *args)
    reported = re.findall(r"(\w*error)=([-+0-9.e]+)", output)
    assert len(reported) == expected, output
    assert all(float(value) < tolerance for _, value in reported), output


def test_torch_shell_fit_recovers_parameters():
    output = run("fem/fitting/torch_shell_material_fit.py")
    assert all(value < 1e-8 for value in errors(output).values()), output
    assert float(re.search(r"fit_difference=([-+0-9.e]+)", output).group(1)) < 1e-10, output
