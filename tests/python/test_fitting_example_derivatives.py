"""Numerical self-checks live here, keeping the runnable examples focused on fitting."""
from litepgo.solver import AbsoluteConvergence

from functools import partial
from pathlib import Path
import runpy

import numpy as np
import pytest
from litepgo import solver

ROOT = Path(__file__).resolve().parents[2] / "examples/fem/fitting"


def check_jacobian(residual, p, h=1e-6, tolerance=1e-6):
    J = residual.evaluate(p).jacobian().toarray()
    fd = np.column_stack([(residual.evaluate(p + delta).value - residual.evaluate(p - delta).value) / (2 * h) for delta in h * np.eye(len(p))])
    assert np.linalg.norm(J - fd) / np.linalg.norm(J) < tolerance
    singular = np.linalg.svd(J, compute_uv=False)
    assert singular[-1] / singular[0] > 1e-6


@pytest.mark.parametrize(
    "case",
    [
        "volume_fixed_pose_fp_fit",
        "shell_fixed_pose_fp_fit",
        "shell_fixed_pose_material_fit",
    ],
)
def test_fixed_pose_derivatives_and_identifiability(case):
    example = runpy.run_path(str(ROOT / f"{case}.py"))
    residual, truth, initial, _ = example["make_problem"]()
    plastic = "_fp_" in case
    p0, target = (np.log(initial), np.log(truth)) if plastic else (initial, truth)
    result = solver.gauss_newton_solve(
        residual, p0, solver.GaussNewtonOptions(gradient_tolerance=1e-12)
    )
    assert result.status == solver.GaussNewtonStatus.Converged
    for p in (p0, target, result.x):
        check_jacobian(residual, p)
    # Target was generated independently through the energy gradient.
    assert np.linalg.norm(residual.evaluate(target).value) < 1e-12
    assert np.linalg.norm(residual.evaluate(result.x).value) < 1e-9


def test_shell_material_equilibrium_derivatives_and_boundary():
    example = runpy.run_path(str(ROOT / "shell_equilibrium_material_fit.py"))
    problem = example["make_problem"]()
    make_energy, make_response = example["make_factories"](problem)
    measure = partial(example["measure"], problem, make_energy, make_response)
    truth = np.array([2.0, 3.0])
    target = measure(truth).reaction
    residual = example["Residual"](measure, target, np.linalg.norm(target))
    for theta in (np.array([0.7, 1.2]), truth):
        check_jacobian(residual, theta, h=1e-4, tolerance=2e-7)
        for load in (1.0, 0.6):
            obs = measure(theta, load=load, need_jacobian=True)
            u = obs.displacement
            fixed, free = problem.fixed.fixed_dofs, problem.fixed.free_dofs
            np.testing.assert_array_equal(u[fixed], load * problem.prescribed[fixed])
            force = make_energy(theta).evaluate(u).gradient()
            assert np.max(np.abs(force[free])) < 2e-12
            np.testing.assert_allclose(
                force, make_response(u).evaluate(theta).value, atol=1e-12
            )
            np.testing.assert_allclose(obs.jacobian @ theta, obs.reaction, atol=1e-11)
            for delta in 1e-4 * np.eye(2):
                fd = (
                    measure(theta + delta, load=load).displacement
                    - measure(theta - delta, load=load).displacement
                ) / 2
                np.testing.assert_allclose(obs.state_jacobian @ delta, fd, atol=1e-10)


def test_volume_material_reaction_scaling_and_derivatives():
    example = runpy.run_path(str(ROOT / "volume_equilibrium_material_fit.py"))
    residual, truth, reaction, _ = example["make_problem"]()
    check_jacobian(residual, np.array([0.2, 0.4]), h=1e-4, tolerance=2e-6)
    for theta in (np.ones(2), truth):
        for axis, strain in ((0, 0.08), (2, -0.04), (2, -0.07)):
            force, derivative = reaction(theta, axis, strain, True)
            np.testing.assert_allclose(derivative @ theta, force, rtol=1e-7, atol=1e-7)
    problem = example["make_geometry"]()
    for axis, strain in ((0, 0.08), (2, -0.04)):
        initial = np.zeros(problem.mapping.num_dofs)
        initial[3 * problem.top + axis] = strain * problem.height
        equilibrium = solver.StaticEquilibriumResponse(
            2,
            problem.fixed,
            initial,
            solver.NewtonOptions(max_iterations=100),
            problem.make_energy,
            problem.make_response,
            convergence=AbsoluteConvergence(1e-10),
        )
        u = equilibrium.evaluate(truth).value
        np.testing.assert_array_equal(
            u[problem.fixed.fixed_dofs], initial[problem.fixed.fixed_dofs]
        )
        assert (
            np.max(
                np.abs(
                    problem.make_energy(truth).evaluate(u).gradient()[problem.fixed.free_dofs]
                )
            )
            < 2e-10
        )


def test_volume_material_cpp_python_recovery():
    import subprocess
    import sys
    from litepgo import _core

    build = Path(_core.__file__).parents[2]
    for command in (
        [str(build / "examples/fem/fitting/volume_equilibrium_material_fit")],
        [sys.executable, str(ROOT / "volume_equilibrium_material_fit.py")],
    ):
        # The native example takes over two minutes in Windows Debug CI.
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        fields = dict(
            part.strip().split("=", 1) for part in completed.stdout.strip().split(",")
        )
        np.testing.assert_allclose(
            np.fromstring(fields["fitted"], sep=" "), [2.0, 3.0], rtol=1e-4
        )
        assert float(fields["parameter error"]) < 1e-4
        assert float(fields["residual"]) < 1e-7
        assert float(fields["held-out error"]) < 1e-4
