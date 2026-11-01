import subprocess
import sys
from pathlib import Path

import numpy as np
from litepgo import _core, mesh


def test_shell_plastic_field_equilibrium_obj(tmp_path):
    root = Path(__file__).resolve().parents[2]
    build = Path(_core.__file__).parents[2]
    outputs = []
    for name, command in (
        ("cpp", [str(build / "example/fem/shell_plastic_field")]),
        ("python", [sys.executable, str(root / "example/fem/shell_plastic_field.py")]),
    ):
        output = tmp_path / name
        run = subprocess.run([*command, str(output)], capture_output=True, text=True, timeout=120)
        assert run.returncode == 0, run.stdout + run.stderr
        rows = [dict(part.split("=") for part in line.split(", ")) for line in run.stdout.splitlines()]
        assert len(rows) == 5
        assert float(rows[0]["max_displacement"]) == 0
        for row in rows:
            assert np.isfinite([float(v) for v in row.values()]).all()
            assert float(row["free_residual"]) < 1e-8
        rest = mesh.read_obj(output / "rest.obj")
        result = mesh.read_obj(output / "equilibrium.obj")
        np.testing.assert_array_equal(rest.cells, result.cells)
        assert result.positions.shape == (65, 3) and result.cells.shape == (96, 3)
        assert np.isfinite(result.positions).all()
        np.testing.assert_allclose(result.positions[:10], rest.positions[:10], atol=1e-14)
        displacement = result.positions-rest.positions
        assert .4 < np.linalg.norm(displacement, axis=1).max() < .5
        np.testing.assert_allclose(displacement[:, 2], 0, atol=1e-12)
        edges1 = result.positions[result.cells[:, 1]] - result.positions[result.cells[:, 0]]
        edges2 = result.positions[result.cells[:, 2]] - result.positions[result.cells[:, 0]]
        assert (np.cross(edges1, edges2)[:, 2] > 0).all()
        outputs.append(result)
    np.testing.assert_array_equal(outputs[0].cells, outputs[1].cells)
    np.testing.assert_allclose(outputs[0].positions, outputs[1].positions, atol=1e-9, rtol=0)
