import runpy
from pathlib import Path
import numpy as np
from litepgo import dof, energy, solver

ROOT = Path(__file__).resolve().parents[2] / "examples"


def test_mesh_response_subclass_assembly_and_solution():
    example = runpy.run_path(str(ROOT / "energy/response/mesh_response_subclass.py"))
    inputs = dof.StencilDofMap(2, [[0, 1]] * 3, 1)
    outputs = dof.StencilDofMap(3, [[0, 1], [1, 2], [0, 2]], 1)
    A = np.array([[1.0, 2.0], [3.0, -1.0]])
    truth = np.array([2.0, 3.0])
    observed = np.tile(A @ truth, (3, 1))

    response = example["LinearResidual"](inputs, outputs, A, observed)
    x = np.array([0.7, 1.2])
    # Independently sum the local rows assigned to each global output.
    J = np.array([[2.0, 4.0], [4.0, 1.0], [6.0, -2.0]])
    np.testing.assert_allclose(response.eval_value(x), J @ (x - truth))
    np.testing.assert_allclose(response.eval_jacobian(x).toarray(), J)
    v, w = np.array([0.3, -0.8]), np.array([0.2, -0.5, 0.7])
    jv, jtw = response.eval_jvp(x, v), response.eval_vjp(x, w)
    np.testing.assert_allclose(jv, J @ v)
    np.testing.assert_allclose(jtw, J.T @ w)
    np.testing.assert_allclose(w @ jv, v @ jtw)
    h = 1e-5
    np.testing.assert_allclose(
        jv,
        (response.eval_value(x + h * v) - response.eval_value(x - h * v)) / (2 * h),
        atol=1e-9,
    )
    result = solver.gauss_newton_solve(
        response, x, solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    )
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, truth, atol=1e-8, rtol=0)


def test_mesh_energy_subclass_assembly_and_solution():
    example = runpy.run_path(str(ROOT / "energy/scalar/mesh_energy_subclass.py"))
    mapping = dof.StencilDofMap(4, [[0, 1], [1, 2], [2, 3]], 1)
    stiffness = np.array([2.0, 3.0, 4.0])
    lengths = np.array([1.0, 2.0, 1.0])

    potential = example["SpringEnergy"](mapping, stiffness, lengths)
    initial = np.array([0.0, 0.0, 0.0, 7.0])
    value, g, H = potential.eval(
        initial, gradient=True, hessian=energy.HessianKind.Exact
    )
    # Independent global incidence-matrix check catches shared-node assembly errors.
    D = np.array([[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 0.0, -1.0, 1.0]])
    stretch = D @ initial - lengths
    np.testing.assert_allclose(value, 0.5 * np.dot(stiffness, stretch**2))
    np.testing.assert_allclose(g, D.T @ (stiffness * stretch))
    np.testing.assert_allclose(H.toarray(), D.T @ np.diag(stiffness) @ D)

    result = solver.newton_solve(
        potential,
        dof.DofReduction(4, [0, 3]),
        initial,
        solver.NewtonOptions(gradient_tolerance=1e-10),
    )
    assert result.status == solver.NewtonStatus.Converged
    # Springs in series have equal tension; extension is distributed as force/k.
    tension = (initial[-1] - lengths.sum()) / (1 / stiffness).sum()
    expected = np.r_[0.0, np.cumsum(lengths + tension / stiffness)]
    np.testing.assert_allclose(result.x, expected, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(result.x[[0, 3]], initial[[0, 3]])
    assert np.max(np.abs(potential.eval_gradient(result.x)[1:3])) < 1e-10
