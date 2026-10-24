import numpy as np
import pytest
import scipy.sparse as sp

from litepgo.energy import AffineMappedEnergy, HessianKind, QuadraticEnergy, WeightedEnergySum
from litepgo.linalg.sparse_solver import AutoIndefiniteSparseSolver, EigenLDLT
from litepgo.solver import NewtonOptions, NewtonStatus, newton_solve


def spd(n):
    return sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")


def test_quadratic_converges_in_one_step():
    A, b = spd(6), np.linspace(-1.0, 2.0, 6)
    result = newton_solve(QuadraticEnergy(A, b), np.zeros(6), NewtonOptions(gradient_tolerance=1e-10))
    assert result.status == NewtonStatus.Converged and result.iterations == 1
    np.testing.assert_allclose(result.x, np.linalg.solve(A.toarray(), -b), atol=1e-12)


def test_fixed_dofs_keep_their_initial_values():
    A, b = spd(5), np.linspace(-1.0, 1.0, 5)
    x0 = np.zeros(5)
    x0[[1, 3]] = [0.5, -2.0]
    result = newton_solve(QuadraticEnergy(A, b), x0, NewtonOptions(gradient_tolerance=1e-12), fixed_dofs=[3, 1])
    assert result.status == NewtonStatus.Converged
    np.testing.assert_array_equal(result.x[[1, 3]], [0.5, -2.0])
    free = [0, 2, 4]
    dense = A.toarray()
    expected = np.linalg.solve(dense[np.ix_(free, free)], -(b[free] + dense[np.ix_(free, [1, 3])] @ [0.5, -2.0]))
    np.testing.assert_allclose(result.x[free], expected, atol=1e-12)


def test_solves_through_composites():
    first, second = QuadraticEnergy(spd(4), np.ones(4)), QuadraticEnergy(sp.identity(4, format="csc"))
    T = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]))
    energy = AffineMappedEnergy(WeightedEnergySum([(first, 1.0), (second, 0.5)]), T)
    result = newton_solve(energy, np.zeros(2), NewtonOptions(gradient_tolerance=1e-12), sparse_solver=EigenLDLT())
    H = T.T @ (spd(4) + 0.5 * sp.identity(4)) @ T
    np.testing.assert_allclose(result.x, np.linalg.solve(H.toarray(), -(T.T @ np.ones(4))), atol=1e-12)


def test_exact_hessian_with_an_indefinite_solver():
    # An indefinite quadratic: Newton must shift its way to a descent direction
    # and then be stopped by the iteration limit, as the energy is unbounded below.
    A = sp.csc_matrix(np.diag([1.0, -1.0]))
    options = NewtonOptions(gradient_tolerance=1e-12, hessian=HessianKind.Exact, max_iterations=3)
    steps = []
    result = newton_solve(QuadraticEnergy(A, np.array([1.0, 1.0])), np.zeros(2), options,
                          sparse_solver=AutoIndefiniteSparseSolver(), on_iteration=steps.append)
    assert result.status == NewtonStatus.MaxIterations
    assert steps[0].diagonal_shifts > 0


def test_callback_steps_outlive_the_solve():
    steps = []
    newton_solve(QuadraticEnergy(spd(3), np.ones(3)), np.zeros(3), NewtonOptions(gradient_tolerance=1e-12),
                 on_iteration=steps.append)
    assert [step.iteration for step in steps] == [0, 1]
    assert steps[0].step == 1.0 and steps[0].line_search_evaluations == 1
    assert steps[-1].gradient_norm <= 1e-12


def test_callback_exceptions_propagate():
    def fail(step):
        raise KeyError("stop")

    with pytest.raises(KeyError):
        newton_solve(QuadraticEnergy(spd(3), np.ones(3)), np.zeros(3), NewtonOptions(gradient_tolerance=1e-12),
                     on_iteration=fail)


def test_options():
    options = NewtonOptions(gradient_tolerance=1e-6)
    assert options.max_iterations == 50 and options.hessian == HessianKind.Projected
    options.max_iterations = 5
    assert options.max_iterations == 5
    with pytest.raises(TypeError):
        NewtonOptions()
    with pytest.raises(ValueError):
        newton_solve(QuadraticEnergy(spd(3)), np.ones(3), NewtonOptions(gradient_tolerance=float("nan")))
