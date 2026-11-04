import pytest

import torch
from litepgo import torch as bridge


@pytest.mark.parametrize(
    "case",
    [
        "volume_equilibrium_material_fit",
        "shell_equilibrium_material_fit",
        "volume_equilibrium_fp_fit",
        "shell_equilibrium_fp_fit",
    ],
)
def test_material_example(case):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    path = root / f"examples/fem/fitting/{case}_torch.py"
    command = [sys.executable, str(path)]
    if case.startswith("volume_"):
        command.append(str(root / "tests/assets/box-tet.veg"))
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    fields = dict(
        part.strip().split("=", 1) for part in completed.stdout.strip().split(",")
    )
    # Recovery checks belong in tests, independently of optimizer stopping criteria.
    if case == "volume_equilibrium_material_fit":
        assert float(fields["parameter error"]) < 1e-4
        assert float(fields["held-out error"]) < 1e-4
    else:
        assert fields["converged"] == "1"
        assert float(fields["parameter_error"]) < 1e-6
        assert float(fields["relative_residual"]) < 1e-8
        assert float(fields["held_out_error"]) < 1e-8


def test_volume_material_equilibrium_end_to_end_gradcheck():
    import runpy
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/fem/fitting/volume_equilibrium_material_fit.py"
    )
    residual, _, _, _ = runpy.run_path(str(path))["make_problem"]()
    q = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        # Backward now re-solves the system; allow solver roundoff between calls.
        lambda q: bridge.response(residual, q),
        (q,),
        eps=1e-4,
        atol=1e-6,
        rtol=1e-4,
        nondet_tol=1e-12,
    )
