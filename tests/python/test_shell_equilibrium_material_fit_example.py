import subprocess
import sys
from pathlib import Path

import numpy as np
from litepgo import _core


def test_shell_equilibrium_material_fit_python_matches_cpp():
    root = Path(__file__).resolve().parents[2]
    build = Path(_core.__file__).parents[2]
    results = []
    for command in (
        [str(build / "examples/fem/fitting/shell_equilibrium_material_fit")],
        [
            sys.executable,
            str(root / "examples/fem/fitting/shell_equilibrium_material_fit.py"),
        ],
    ):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        fields = dict(
            part.strip().split("=", 1) for part in completed.stdout.strip().split(",")
        )
        values = {key: np.fromstring(value, sep=" ") for key, value in fields.items()}
        np.testing.assert_allclose(values["fitted"], [2.0, 3.0], rtol=1e-6)
        assert values["converged"].item() == 1
        assert values["parameter_error"].item() < 1e-6
        assert values["relative_residual"].item() < 1e-8
        assert values["held_out_error"].item() < 1e-8
        results.append(values)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        np.testing.assert_allclose(
            results[0][key], results[1][key], rtol=1e-6, atol=1e-9
        )
