import numpy as np
import pytest
import scipy.sparse as sp

from litepgo.energy import FixedPatternResponse
from litepgo.dof import DofReduction
from litepgo.energy import AffineMappedEnergy, HessianKind, QuadraticEnergy, WeightedSumEnergy
from litepgo.linalg.sparse_solver import AutoIndefiniteSparseSolver, EigenLDLT
from litepgo.solver import NewtonOptions, NewtonStatus, newton_solve


class TestResponse(FixedPatternResponse):
    __test__ = False

    def __init__(self, pattern, evaluator):
        super().__init__(pattern)
        self.evaluator = evaluator

    def eval_impl(self, x, need_jacobian):
        return self.evaluator(x, need_jacobian)


def free(n):
    return DofReduction(n, [])


def spd(n):
    return sp.diags([-1.0, 4.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")


def test_quadratic_converges_in_one_step():
    A, b = spd(6), np.linspace(-1.0, 2.0, 6)
    result = newton_solve(QuadraticEnergy(A, b), free(6), np.zeros(6), NewtonOptions(gradient_tolerance=1e-10))
    assert result.status == NewtonStatus.Converged and result.iterations == 1
    np.testing.assert_allclose(result.x, np.linalg.solve(A.toarray(), -b), atol=1e-12)


def test_fixed_dofs_keep_their_initial_values():
    A, b = spd(5), np.linspace(-1.0, 1.0, 5)
    x0 = np.zeros(5)
    x0[[1, 3]] = [0.5, -2.0]
    result = newton_solve(QuadraticEnergy(A, b), DofReduction(5, [3, 1]), x0, NewtonOptions(gradient_tolerance=1e-12))
    assert result.status == NewtonStatus.Converged
    np.testing.assert_array_equal(result.x[[1, 3]], [0.5, -2.0])
    free = [0, 2, 4]
    dense = A.toarray()
    expected = np.linalg.solve(dense[np.ix_(free, free)], -(b[free] + dense[np.ix_(free, [1, 3])] @ [0.5, -2.0]))
    np.testing.assert_allclose(result.x[free], expected, atol=1e-12)


def test_solves_through_composites():
    first, second = QuadraticEnergy(spd(4), np.ones(4)), QuadraticEnergy(sp.identity(4, format="csc"))
    T = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]))
    energy = AffineMappedEnergy(WeightedSumEnergy([(first, 1.0), (second, 0.5)]), T)
    result = newton_solve(energy, free(2), np.zeros(2), NewtonOptions(gradient_tolerance=1e-12), EigenLDLT())
    H = T.T @ (spd(4) + 0.5 * sp.identity(4)) @ T
    np.testing.assert_allclose(result.x, np.linalg.solve(H.toarray(), -(T.T @ np.ones(4))), atol=1e-12)


def test_exact_hessian_with_an_indefinite_solver():
    # An indefinite quadratic: Newton must shift its way to a descent direction
    # and then be stopped by the iteration limit, as the energy is unbounded below.
    A = sp.csc_matrix(np.diag([1.0, -1.0]))
    options = NewtonOptions(gradient_tolerance=1e-12, hessian=HessianKind.Exact, max_iterations=3)
    steps = []
    result = newton_solve(QuadraticEnergy(A, np.array([1.0, 1.0])), free(2), np.zeros(2), options,
                          AutoIndefiniteSparseSolver(), on_iteration=steps.append)
    assert result.status == NewtonStatus.MaxIterations
    assert steps[0].diagonal_shifts > 0


def test_callback_steps_outlive_the_solve():
    steps = []
    newton_solve(QuadraticEnergy(spd(3), np.ones(3)), free(3), np.zeros(3), NewtonOptions(gradient_tolerance=1e-12),
                 on_iteration=steps.append)
    assert [step.iteration for step in steps] == [0, 1]
    assert steps[0].step == 1.0 and steps[0].line_search_evaluations == 1
    assert steps[-1].gradient_norm <= 1e-12


def test_callback_exceptions_propagate():
    def fail(step):
        raise KeyError("stop")

    with pytest.raises(KeyError):
        newton_solve(QuadraticEnergy(spd(3), np.ones(3)), free(3), np.zeros(3), NewtonOptions(gradient_tolerance=1e-12),
                     on_iteration=fail)


def test_options():
    options = NewtonOptions(gradient_tolerance=1e-6)
    assert options.max_iterations == 50 and options.hessian == HessianKind.Projected
    options.max_iterations = 5
    assert options.max_iterations == 5
    # Like the C++ aggregate, the tolerance defaults to NaN and must be set before solving.
    defaults = NewtonOptions()
    assert np.isnan(defaults.gradient_tolerance)
    defaults.gradient_tolerance = 1e-8
    assert defaults.gradient_tolerance == 1e-8
    with pytest.raises(ValueError):
        newton_solve(QuadraticEnergy(spd(3)), free(3), np.ones(3), NewtonOptions())


def test_gauss_newton_analytic_callback_and_owned_inputs():
    from litepgo.solver import GaussNewtonOptions, GaussNewtonStatus, gauss_newton_solve

    calls = []
    A = np.array([[1., 0.], [0., 1.], [1., 0.]])
    def residual(x, need_jacobian):
        calls.append((x, x.copy(), need_jacobian))
        r = A @ x - [1., 2., 3.]
        return (r, A) if need_jacobian else r

    result = gauss_newton_solve(TestResponse(np.ones((3, 2)), residual), np.zeros(2), GaussNewtonOptions(gradient_tolerance=1e-8))
    assert result.status == GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, [2., 2.], atol=1e-8)
    assert result.objective == pytest.approx(1.)
    assert result.gradient_norm <= 1e-8
    assert result.evaluations == len(calls)
    assert {need for _, _, need in calls} == {True, False}
    for x, snapshot, _ in calls:
        np.testing.assert_array_equal(x, snapshot)


def test_gauss_newton_infeasible_trials_and_errors():
    from litepgo.solver import GaussNewtonOptions, GaussNewtonStatus, gauss_newton_solve

    rejected = []
    def residual(x, need_jacobian):
        if x[0] <= 0:
            rejected.append(x[0])
            return None
        r = np.log(x)
        return (r, np.array([[1 / x[0]]])) if need_jacobian else r

    options = GaussNewtonOptions(gradient_tolerance=1e-9)
    result = gauss_newton_solve(TestResponse(np.ones((1, 1)), residual), np.array([10.]), options)
    assert result.status == GaussNewtonStatus.Converged and rejected
    np.testing.assert_allclose(result.x, [1.], atol=1e-8)
    with pytest.raises(ValueError):
        gauss_newton_solve(TestResponse(np.ones((1, 1)), residual), np.array([-1.]), options)

    def fail(x, need_jacobian):
        raise KeyError("callback bug")
    with pytest.raises(KeyError, match="callback bug"):
        gauss_newton_solve(TestResponse(np.ones((1, 1)), fail), np.ones(1), options)
    with pytest.raises(TypeError, match="Expected"):
        gauss_newton_solve(TestResponse(np.ones((1, 1)), lambda x, need: x), np.ones(1), options)
    with pytest.raises(RuntimeError, match="invalid Jacobian"):
        gauss_newton_solve(TestResponse(np.ones((1, 1)), lambda x, need: (x, np.zeros((2, 2)))), np.ones(1), options)
    assert np.isnan(GaussNewtonOptions().gradient_tolerance)


def test_response_pattern_and_object_protocol():
    from litepgo.energy import Response
    from litepgo.solver import gauss_newton_solve, GaussNewtonOptions

    r = TestResponse(sp.eye(2, format="csc"),
                             lambda x, need: (x*x, np.diag(2*x)) if need else x*x)
    assert isinstance(r, Response)
    np.testing.assert_array_equal(r.evaluate(np.array([2.0, 3.0])).value, [4.0, 9.0])
    J = r.evaluate(np.zeros(2)).jacobian()
    assert J.nnz == 2  # Numerical zeros do not remove structural slots.
    with pytest.raises(ValueError):
        r.evaluate(np.zeros(3)).value
    bad = TestResponse(sp.eye(2), lambda x, need: (x, np.ones((2, 2))))
    with pytest.raises(RuntimeError, match="outside pattern"):
        bad.evaluate(np.ones(2)).jacobian()
    with pytest.raises(TypeError):
        gauss_newton_solve(lambda x, need: x, np.ones(1), GaussNewtonOptions(gradient_tolerance=1e-8))


def test_fixed_response_requires_override():
    base = FixedPatternResponse(np.ones((1, 1)))
    with pytest.raises(RuntimeError, match="eval_impl"):
        base.evaluate(np.ones(1)).value
    with pytest.raises(TypeError):
        FixedPatternResponse(np.ones((1, 1)), lambda x, need: x)


def test_gauss_newton_explicit_sparse_solver():
    from litepgo.solver import gauss_newton_solve, GaussNewtonOptions, GaussNewtonStatus
    r = TestResponse(np.eye(2), lambda x, need: (x - 1, np.eye(2)) if need else x - 1)
    result = gauss_newton_solve(r, np.zeros(2), GaussNewtonOptions(gradient_tolerance=1e-9), solver=EigenLDLT())
    assert result.status == GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, np.ones(2), atol=1e-8)


@pytest.mark.parametrize('maximum', [.1, np.inf])
@pytest.mark.parametrize('ccd_bound', [.5, 1.])
def test_newton_coordinate_limit_precedes_ccd_without_extra_hessian(maximum, ccd_bound):
    from litepgo.energy import FixedPatternEnergy

    class CountingQuadratic(FixedPatternEnergy):
        def __init__(self):
            super().__init__(sp.eye(2, format='csc'))
            self.hessians = 0
            self.feasibility_queries = []

        def eval_values(self, x, want_gradient, kind):
            r = x - [1., -.4]
            self.hessians += kind is not None
            return .5*r @ r, r if want_gradient else None, np.eye(2) if kind is not None else None

        def feasible_step(self, x, direction):
            self.feasibility_queries.append((x.copy(), direction.copy()))
            return ccd_bound

    potential = CountingQuadratic()
    steps = []
    options = NewtonOptions(gradient_tolerance=1e-12, max_iterations=1, max_coordinate_step=maximum)
    result = newton_solve(potential, free(2), np.zeros(2), options, on_iteration=steps.append)
    scale = min(1., maximum)
    np.testing.assert_array_equal(potential.feasibility_queries[0][1], np.zeros(2))
    np.testing.assert_allclose(potential.feasibility_queries[1][1], scale*np.array([1., -.4]))
    np.testing.assert_allclose(result.x, ccd_bound*scale*np.array([1., -.4]))
    assert steps[0].feasible_step == steps[0].step == ccd_bound*scale
    assert potential.hessians == result.iterations + 1


@pytest.mark.parametrize('maximum', [0., -1., -np.inf, np.nan])
def test_newton_rejects_invalid_coordinate_limit(maximum):
    with pytest.raises(ValueError, match='max_coordinate_step'):
        newton_solve(QuadraticEnergy(sp.eye(2, format='csc')), free(2), np.ones(2),
                     NewtonOptions(gradient_tolerance=1e-10, max_coordinate_step=maximum))


@pytest.mark.parametrize('fixed', [[], [0, 1]])
def test_newton_checks_feasibility_before_stationary_or_fixed_return(fixed):
    from litepgo.energy import FixedPatternEnergy

    class InvalidStationaryState(FixedPatternEnergy):
        def __init__(self):
            super().__init__(sp.eye(2, format='csc'))

        def eval_values(self, x, want_gradient, kind):
            pytest.fail('initial feasibility must be checked before evaluation')

        def feasible_step(self, x, direction):
            np.testing.assert_array_equal(direction, np.zeros(2))
            raise ValueError('invalid initial geometry')

    with pytest.raises(ValueError, match='invalid initial geometry'):
        newton_solve(InvalidStationaryState(), DofReduction(2, fixed), np.zeros(2),
                     NewtonOptions(gradient_tolerance=1e-10))
