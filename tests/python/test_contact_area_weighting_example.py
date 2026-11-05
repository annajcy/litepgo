import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from litepgo import _core

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples/contact"


def test_cpp_python_area_weighting_example(tmp_path):
    executable = Path(_core.__file__).parents[2] / "examples/contact/contact_area_weighting"
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    outputs = []
    for name, command in (("cpp", [str(executable)]),
                          ("python", [sys.executable, str(EXAMPLES / "contact_area_weighting.py")])):
        output = tmp_path / f"{name}.csv"
        run = subprocess.run([*command, str(output)], capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
        data = np.genfromtxt(output, delimiter=",", names=True)
        assert data.shape == (6,)
        for weighted in (0, 1):
            rows = data[data["use_area_weighting"] == weighted]
            np.testing.assert_array_equal(rows["subdivisions"], [4, 8, 16])
            np.testing.assert_array_equal(rows["triangles"], [64, 256, 1024])
            np.testing.assert_allclose(rows["edge_length"], [.04, .02, .01])
            for key in ("energy", "normal_force"):
                assert np.isfinite(rows[key]).all() and np.all(rows[key] > 0)
                ratio = rows[key][-1] / rows[key][0]
                if weighted:
                    assert .8 < ratio < 1.3
                else:
                    assert ratio > 10
        outputs.append(data)
    assert outputs[0].dtype.names == outputs[1].dtype.names
    for key in outputs[0].dtype.names:
        np.testing.assert_allclose(outputs[0][key], outputs[1][key], rtol=1e-11, atol=0)


def test_area_weighting_plot(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.image import imread

    data = tmp_path / "comparison.csv"
    subprocess.run([sys.executable, str(EXAMPLES / "contact_area_weighting.py"), str(data)],
                   check=True, capture_output=True, text=True, timeout=30)
    run = subprocess.run([sys.executable, str(EXAMPLES / "plot_contact_area_weighting.py"), str(data)],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    pixels = imread(data.with_suffix(".png"))
    assert pixels.shape == (720, 1800, 4)
    assert np.isfinite(pixels).all() and np.ptp(pixels[:, :, :3]) > .5
