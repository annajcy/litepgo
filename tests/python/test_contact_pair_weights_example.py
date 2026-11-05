import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from litepgo import _core

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/contact"


def test_cpp_python_group_weights_example_and_plot(tmp_path):
    executable = Path(_core.__file__).parents[2] / "examples/contact/contact_pair_weights"
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    results = []
    for name, command in (("cpp", [str(executable)]),
                          ("python", [sys.executable, str(EXAMPLES / "contact_pair_weights.py")])):
        output = tmp_path / f"{name}.csv"
        run = subprocess.run([*command, str(output)], capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
        data = np.genfromtxt(output, delimiter=",", names=True)
        assert data.shape == (3,)
        np.testing.assert_array_equal(data["multiplier"], [.5, 2, 0])
        np.testing.assert_array_equal(data["active_records"], [1, 1, 0])
        for key in ("energy", "force"):
            np.testing.assert_allclose(data[f"weighted_{key}"] / data[f"reference_{key}"], [.5, 2, 0], rtol=1e-12)
        results.append(data)
    assert results[0].dtype.names == results[1].dtype.names
    for key in results[0].dtype.names:
        np.testing.assert_allclose(results[0][key], results[1][key], rtol=1e-12)
    pytest.importorskip("matplotlib")
    from matplotlib.image import imread

    run = subprocess.run([sys.executable, str(EXAMPLES / "plot_contact_pair_weights.py"), str(tmp_path / "python.csv")],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    pixels = imread(tmp_path / "python.png")
    assert pixels.shape == (720, 1620, 4)
    assert np.isfinite(pixels).all() and np.ptp(pixels[:, :, :3]) > .5
