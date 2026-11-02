import pytest

torch = pytest.importorskip("torch")
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

    path = Path(__file__).resolve().parents[2] / f"examples/fem/fitting/{case}_torch.py"
    subprocess.run([sys.executable, str(path)], check=True, timeout=120)


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
        lambda q: bridge.response(residual, q), (q,), eps=1e-4, atol=1e-6, rtol=1e-4
    )
