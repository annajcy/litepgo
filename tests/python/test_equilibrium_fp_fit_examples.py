import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from litepgo import _core


@pytest.mark.parametrize("kind,truth", [("volume", [1.04, .97, 1.02]), ("shell", [1.06, .95])])
def test_equilibrium_fp_fit_python_matches_cpp(kind, truth):
    root = Path(__file__).resolve().parents[2]
    build = Path(_core.__file__).parents[2]
    name = f"{kind}_equilibrium_fp_fit"
    outputs = []
    for command in ([str(build / "example/fem/fitting" / name)],
                    [sys.executable, str(root / "example/fem/fitting" / f"{name}.py")]):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        fields = dict(part.strip().split("=", 1) for part in completed.stdout.strip().split(","))
        values = {key: np.fromstring(value, sep=" ") for key, value in fields.items()}
        assert all(np.isfinite(value).all() for value in values.values())
        np.testing.assert_allclose(values["truth"], truth, rtol=0, atol=1e-12)
        np.testing.assert_allclose(values["fitted"], truth, rtol=1e-6, atol=0)
        assert values["converged"].item() == 1
        assert values["parameter_error"].item() < 1e-6
        assert values["relative_residual"].item() < 1e-8
        assert values["held_out_error"].item() < 1e-8
        for key in ("state_fd_error", "reaction_fd_error"):
            assert values[key].item() < 2e-6
        for key in ("initial_ratio", "truth_ratio", "fitted_ratio"):
            assert 1e-6 < values[key].item() <= 1
        outputs.append(values)
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], rtol=1e-6, atol=1e-8)
