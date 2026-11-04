import subprocess
import sys
from pathlib import Path

import numpy as np
from litepgo import _core


def test_volume_fixed_pose_fp_fit_python_matches_cpp():
    root = Path(__file__).resolve().parents[2]
    build = Path(_core.__file__).parents[2]
    results = []
    for command in (
        [str(build / "examples/fem/fitting/volume_fixed_pose_fp_fit")],
        [
            sys.executable,
            str(root / "examples/fem/fitting/volume_fixed_pose_fp_fit.py"),
        ],
    ):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        fields = dict(
            part.strip().split("=", 1) for part in completed.stdout.strip().split(",")
        )
        values = {key: np.fromstring(value, sep=" ") for key, value in fields.items()}
        assert all(np.isfinite(value).all() for value in values.values())
        np.testing.assert_array_equal(values["truth"], [1.08, 0.94, 1.03])
        np.testing.assert_array_equal(values["initial"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(values["fitted"], values["truth"], rtol=1e-7, atol=0)
        assert (values["fitted"] > 0).all()
        assert values["converged"].item() == 1
        assert 0 < values["iterations"].item() < 50
        assert values["parameter_error"].item() < 1e-7
        assert values["relative_residual"].item() < 1e-9
        results.append(values)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        np.testing.assert_allclose(
            results[0][key], results[1][key], rtol=1e-6, atol=1e-10
        )
