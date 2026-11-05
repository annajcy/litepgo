import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from litepgo import _core, mesh

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/contact"


def test_cpp_python_snapshot_example_and_plot(tmp_path):
    executable = Path(_core.__file__).parents[2] / "examples/contact/contact_snapshot"
    if sys.platform == "win32":
        executable = executable.with_suffix(".exe")
    outputs = []
    for name, command in (("cpp", [str(executable)]),
                          ("python", [sys.executable, str(EXAMPLES / "contact_snapshot.py")])):
        output = tmp_path / name
        run = subprocess.run([*command, str(output)], capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
        with (output / "contacts.csv").open(newline="") as stream:
            data = list(csv.DictReader(stream))
        assert len(data) == 20
        assert {row["type"] for row in data} == {"VV", "VE", "VF", "EE"}
        assert sum(float(row["mollifier"]) == 0 for row in data) == 1
        surface = mesh.read_obj(output / "mesh.obj")
        assert surface.positions.shape == (30, 3) and surface.cells.shape == (10, 3)
        outputs.append((surface, data))
    np.testing.assert_array_equal(outputs[0][0].positions, outputs[1][0].positions)
    np.testing.assert_array_equal(outputs[0][0].cells, outputs[1][0].cells)
    for cpp, py in zip(outputs[0][1], outputs[1][1]):
        assert cpp["type"] == py["type"]
        np.testing.assert_allclose([float(cpp[key]) for key in cpp if key != "type"],
                                   [float(py[key]) for key in py if key != "type"], rtol=1e-12)

    pytest.importorskip("matplotlib")
    from matplotlib.image import imread

    run = subprocess.run([sys.executable, str(EXAMPLES / "plot_contact_snapshot.py"), str(tmp_path / "python")],
                         capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    for filename in ("contacts.png", "contacts_details.png"):
        pixels = imread(tmp_path / "python" / filename)
        assert pixels.shape == (1280, 1920, 4)
        assert np.isfinite(pixels).all() and np.ptp(pixels[:, :, :3]) > .5


def test_plot_closest_points_on_mollified_edge_endpoints():
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("snapshot_plot", EXAMPLES / "plot_contact_snapshot.py")
    plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot)
    # Near-parallel endpoint/edge, parallel overlapping, and disjoint endpoint/endpoint.
    for points, expected in (
        ([[-1, 0, 0], [1, 0, 0], [0, .004, .2], [2, .008, .2]], [[0, 0, 0], [0, .004, .2]]),
        ([[-1, 0, 0], [1, 0, 0], [-.7, 0, .2], [.8, 0, .2]], [[-.7, 0, 0], [-.7, 0, .2]]),
        ([[0, 0, 0], [1, 0, 0], [2, 0, .2], [3, 0, .2]], [[1, 0, 0], [2, 0, .2]]),
    ):
        np.testing.assert_allclose(plot.closest_points("EE", np.array(points, dtype=float)), expected, atol=1e-14)
