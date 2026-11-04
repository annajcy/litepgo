import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from litepgo import _core


@pytest.mark.parametrize(
    "kind,truth", [("volume", [1.04, 0.97, 1.02]), ("shell", [1.06, 0.95])]
)
def test_equilibrium_fp_fit_python_matches_cpp(kind, truth):
    root = Path(__file__).resolve().parents[2]
    build = Path(_core.__file__).parents[2]
    name = f"{kind}_equilibrium_fp_fit"
    outputs = []
    for command in (
        [str(build / "examples/fem/fitting" / name)],
        [sys.executable, str(root / "examples/fem/fitting" / f"{name}.py")],
    ):
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        fields = dict(
            part.strip().split("=", 1) for part in completed.stdout.strip().split(",")
        )
        values = {key: np.fromstring(value, sep=" ") for key, value in fields.items()}
        assert all(np.isfinite(value).all() for value in values.values())
        np.testing.assert_allclose(values["truth"], truth, rtol=0, atol=1e-12)
        np.testing.assert_allclose(values["fitted"], truth, rtol=1e-6, atol=0)
        assert values["converged"].item() == 1
        assert values["parameter_error"].item() < 1e-6
        assert values["relative_residual"].item() < 1e-8
        assert values["held_out_error"].item() < 1e-8
        outputs.append(values)
    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        np.testing.assert_allclose(
            outputs[0][key], outputs[1][key], rtol=1e-6, atol=1e-8
        )


@pytest.fixture(scope="module", params=["volume", "shell"])
def plastic_experiment(request):
    import importlib.util
    from functools import partial

    path = (
        Path(__file__).resolve().parents[2]
        / f"examples/fem/fitting/{request.param}_equilibrium_fp_fit.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"{request.param}_equilibrium_example", path
    )
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    problem = example.make_problem()
    make_energy, make_response = example.make_factories(problem)
    measurement = partial(example.measure, problem, make_energy, make_response)
    truth = np.array([1.04, 0.97, 1.02] if request.param == "volume" else [1.06, 0.95])
    target, _ = measurement(np.log(truth))
    num_free = len(problem.fixed.free_dofs)
    scales = np.concatenate(
        [
            np.full(num_free, max(np.linalg.norm(target[:num_free]), 1e-8)),
            np.full(
                len(target) - num_free, max(np.linalg.norm(target[num_free:]), 1e-8)
            ),
        ]
    )
    residual = example.Residual(measurement, target, scales)
    yield problem, make_energy, make_response, measurement, residual, truth, scales
    del sys.modules[spec.name]


def test_plastic_equilibrium_derivatives_and_boundary_conditions(plastic_experiment):
    from litepgo import solver

    problem, make_energy, make_response, measurement, _, truth, scales = (
        plastic_experiment
    )
    free_ids, fixed_ids = problem.fixed.free_dofs, problem.fixed.fixed_dofs
    for p in (np.zeros(len(truth)), np.log(truth)):
        _, jacobian = measurement(p, need_jacobian=True)
        assert np.linalg.norm(jacobian[: len(free_ids)]) > 1e-5
        # Both displacement and reaction derivatives include the re-solved state.
        for h in (1e-4, 1e-5) if len(truth) == 3 else (3e-5, 1e-5):
            for c in range(len(truth)):
                delta = np.eye(len(truth))[c] * h
                forward, _ = measurement(p + delta)
                backward, _ = measurement(p - delta)
                error = ((forward - backward) / (2 * h) - jacobian[:, c]) / scales
                assert np.max(np.abs(error[: len(free_ids)])) < 2e-6
                assert np.max(np.abs(error[len(free_ids) :])) < 2e-6
        for load in (1.0, 0.6):
            initial = load * problem.prescribed
            equilibrium = solver.StaticEquilibriumResponse(
                len(truth),
                problem.fixed,
                initial,
                solver.NewtonOptions(
                    gradient_tolerance=1e-11 if len(truth) == 3 else 1e-13,
                    max_iterations=100,
                ),
                make_energy,
                make_response,
            )
            displacement = equilibrium.eval_value(p)
            gradient = make_energy(p).eval_gradient(displacement)
            np.testing.assert_array_equal(displacement[fixed_ids], initial[fixed_ids])
            assert np.max(np.abs(gradient[free_ids])) < 2e-11
            np.testing.assert_allclose(
                make_response(displacement).eval_value(p), gradient, atol=1e-12
            )
            observed, no_jacobian = measurement(p, load=load)
            assert no_jacobian is None
            np.testing.assert_allclose(
                observed,
                np.concatenate([displacement[free_ids], gradient[fixed_ids]]),
                atol=1e-12,
            )


def test_plastic_equilibrium_fit_is_identifiable(plastic_experiment):
    from litepgo import solver

    _, _, _, measurement, residual, truth, scales = plastic_experiment
    result = solver.gauss_newton_solve(
        residual,
        np.zeros(len(truth)),
        solver.GaussNewtonOptions(gradient_tolerance=1e-10),
    )
    assert result.status == solver.GaussNewtonStatus.Converged
    for p in (np.zeros(len(truth)), np.log(truth), result.x):
        singular = np.linalg.svd(residual.eval_jacobian(p).toarray(), compute_uv=False)
        assert singular[-1] / singular[0] > 1e-6
    assert np.linalg.norm(np.exp(result.x) - truth) / np.linalg.norm(truth) < 1e-6
    assert np.linalg.norm(residual.eval_value(result.x)) < 1e-8
    held_out, _ = measurement(np.log(truth), load=0.6)
    prediction, _ = measurement(result.x, load=0.6)
    assert np.linalg.norm((prediction - held_out) / scales) < 1e-8
