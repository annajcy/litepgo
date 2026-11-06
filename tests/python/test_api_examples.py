import subprocess
import sys
from pathlib import Path

import pytest
from litepgo import _core


def test_thread_limit_example():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "examples/solver/thread_limit.py")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("equilibrium=") == 3
    assert "previous TBB policy restored" in result.stdout


@pytest.mark.parametrize(
    "folder,name",
    [
        ("fem/basis", "tet_p1_basis"),
        ("fem/basis", "hermite_interpolation"),
        ("fem/quadrature", "quadrature_integration"),
        ("fem/response", "response_hessian"),
        ("energy/scalar", "custom_energy"),
        ("energy/scalar", "mesh_energy_subclass"),
        ("energy/response", "custom_response"),
        ("energy/response", "mesh_response"),
        ("energy/response", "mesh_response_subclass"),
    ],
)
def test_cpp_python_api_examples(folder, name):
    root = Path(__file__).resolve().parents[2]
    build = Path(_core.__file__).parents[2]
    outputs = []
    for command in (
        [str(build / "examples" / folder / name)],
        [sys.executable, str(root / "examples" / folder / f"{name}.py")],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]


@pytest.fixture(scope="module")
def api_modules():
    import importlib.util

    root = Path(__file__).resolve().parents[2] / "examples"
    modules = []
    for folder, name in [
        ("energy/scalar", "custom_energy"),
        ("energy/response", "custom_response"),
    ]:
        spec = importlib.util.spec_from_file_location(
            name, root / folder / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@pytest.mark.parametrize(
    "x,value,g,H",
    [
        ([1.0, 1.0], 5.0, [9.0, 9.0], [[13.0, 12.0], [12.0, 13.0]]),
        ([2.0, -1.0], 2.75, [3.0, 0.0], [[4.0, 3.0], [3.0, 4.0]]),
        ([-2.0, 1.0], 2.5, [-2.0, 1.0], [[1.0, 0.0], [0.0, 1.0]]),
        ([1.0, -1.0], 1.0, [1.0, -1.0], [[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_energy_example_independent_analytic_values(api_modules, x, value, g, H):
    import numpy as np
    from litepgo import energy

    obj = api_modules[0].Coupled()
    x = np.array(x)
    assert obj.eval_value(x) == value
    np.testing.assert_array_equal(obj.eval_gradient(x), g)
    obj.update_pattern(x)
    for kind in (energy.HessianKind.Exact, energy.HessianKind.Projected):
        for want_g in (False, True):
            actual, gradient, hessian = obj.eval(x, gradient=want_g, hessian=kind)
            assert actual == value
            if want_g:
                np.testing.assert_array_equal(gradient, g)
            else:
                assert gradient is None
            np.testing.assert_array_equal(hessian.toarray(), H)
    # Fixed energy has a different target: independently check all outputs.
    fixed = api_modules[0].Quadratic()
    delta = x - [1.0, 2.0]
    assert fixed.eval_value(x) == 0.5 * delta @ delta
    np.testing.assert_array_equal(fixed.eval_gradient(x), delta)
    np.testing.assert_array_equal(fixed.eval_hessian(x).toarray(), np.eye(2))


@pytest.mark.parametrize(
    "x,value,J",
    [
        ([2.0, -1.0], [1.0, 1.0], [[1.0, 0.0], [4.0, 1.0]]),
        ([-2.0, 1.0], [-3.0, -1.0], [[1.0, 0.0], [0.0, 1.0]]),
        ([0.0, 3.0], [-1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]]),
    ],
)
def test_response_example_independent_analytic_values(api_modules, x, value, J):
    import numpy as np

    obj = api_modules[1].NonlinearResidual()
    x = np.array(x)
    np.testing.assert_array_equal(obj.eval_value(x), value)
    obj.update_patterns(x)
    actual, jacobian = obj.eval_value_and_jacobian(x)
    np.testing.assert_array_equal(actual, value)
    np.testing.assert_array_equal(jacobian.toarray(), J)
    fixed = api_modules[1].LinearResidual()
    np.testing.assert_array_equal(
        fixed.eval_jacobian(x).toarray(), [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    np.testing.assert_array_equal(
        fixed.eval_value(x), [x[0] - 1, x[1] - 2, x[0] + x[1] - 3]
    )


def test_example_pattern_errors_and_zeroing(api_modules):
    import numpy as np

    obj = api_modules[0].Coupled()
    with pytest.raises(RuntimeError, match="outside pattern"):
        obj.eval_hessian(np.ones(2))
    assert obj.update_pattern(np.ones(2))
    for x in (-np.ones(2), np.zeros(2)):
        np.testing.assert_array_equal(obj.eval_hessian(x).toarray(), np.eye(2))
    assert obj.update_pattern(-np.ones(2))
    assert not obj.update_pattern(-np.ones(2))
    with pytest.raises(RuntimeError, match="outside pattern"):
        obj.eval_hessian(np.ones(2))

    response = api_modules[1].NonlinearResidual()
    for x in (np.ones(2), -np.ones(2), np.zeros(2), np.ones(2)):
        expected = np.array([[1., 0.], [2 * max(0., x[0]), 1.]])
        np.testing.assert_array_equal(response.eval_jacobian(x).toarray(), expected)
        assert not response.update_patterns(x)


def test_custom_response_example_fit(api_modules):
    import numpy as np
    from litepgo import solver

    result = solver.gauss_newton_solve(
        api_modules[1].NonlinearResidual(),
        -np.ones(2),
        solver.GaussNewtonOptions(gradient_tolerance=1e-9),
    )
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, np.ones(2), atol=1e-8)
