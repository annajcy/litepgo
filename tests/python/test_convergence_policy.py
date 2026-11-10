"""Two-stage convergence snapshots and callback ownership."""
import gc
import weakref
import numpy as np
import pytest
import scipy.sparse as sp
from litepgo import dof, energy, solver


class Quadratic(energy.FixedPatternEnergy):
    def __init__(self):
        super().__init__(sp.eye(2, format='csc'))
        self.hessians = 0

    def eval_values(self, x, need_gradient, need_hessian):
        self.hessians += int(need_hessian)
        return (.5 * x @ x, x.copy() if need_gradient else None,
                sp.eye(2, format='csc') if need_hessian else None)


def solve(potential, policy, **options):
    return solver.newton_solve(potential, dof.DofReduction(2, []), np.ones(2),
                               solver.NewtonOptions(project_hessian=False, **options), convergence=policy)


def test_two_phases_are_owned_readonly_snapshots():
    potential = Quadratic()
    states = []
    def policy(state):
        states.append(state)
        return state.gradient_norm == 0
    result = solve(potential, policy)
    assert result.status == solver.NewtonStatus.Converged
    assert [s.iteration for s in states] == [0, 0, 1]
    assert [s.newton_correction_norm for s in states] == [None, 1., None]
    assert states[0].diagonal_shifts is None and states[0].linear_backward_error is None
    assert states[1].diagonal_shifts == 0 and states[1].linear_backward_error == 0
    assert states[0].energy == states[1].energy == 1 and states[2].energy == 0
    assert not states[1].project_hessian
    assert potential.hessians == 1
    with pytest.raises(AttributeError):
        states[0].energy = 4


def test_phase_one_can_skip_hessian_and_phase_two_can_accept_without_moving():
    potential = Quadratic()
    assert solve(potential, lambda s: True).iterations == 0
    assert potential.hessians == 0
    result = solve(potential, lambda s: s.newton_correction_norm is not None)
    assert result.status == solver.NewtonStatus.Converged and result.iterations == 0
    np.testing.assert_array_equal(result.x, np.ones(2))
    assert potential.hessians == 1


def test_policy_rejection_and_exception_are_not_convergence():
    assert solve(Quadratic(), lambda s: False, max_iterations=0).status == solver.NewtonStatus.MaxIterations
    assert solve(Quadratic(), lambda s: False).status == solver.NewtonStatus.NoDescentDirection
    def broken(state):
        raise RuntimeError('policy failure')
    with pytest.raises(RuntimeError, match='policy failure'):
        solve(Quadratic(), broken)
    with pytest.raises(TypeError, match='callable'):
        solve(Quadratic(), 2)
    r = solver.newton_solve(Quadratic(), dof.DofReduction(2, [0, 1]), np.ones(2),
                           solver.NewtonOptions(), convergence=lambda s: False)
    assert r.status != solver.NewtonStatus.Converged


@pytest.mark.parametrize('tolerance', [-1, np.nan, np.inf, -np.inf])
def test_absolute_tolerance_is_validated(tolerance):
    with pytest.raises(ValueError):
        solver.AbsoluteConvergence(tolerance)


def test_static_response_policy_is_traced_by_gc():
    holder = []
    def policy(state, holder=holder):
        assert holder
        return state.gradient_norm <= 1e-12
    reference = weakref.ref(policy)
    response = solver.StaticEquilibriumResponse(2, dof.DofReduction(2, []), np.zeros(2),
        solver.NewtonOptions(), lambda p: energy.QuadraticEnergy(sp.eye(2, format='csc'), -p),
        lambda u: energy.AffineResponse(-sp.eye(2, format='csc'), u), convergence=policy)
    holder.append(response)
    evaluation = response.evaluate(np.zeros(2))
    np.testing.assert_allclose(evaluation.jacobian().toarray(), np.eye(2))
    del response, evaluation, policy, holder
    gc.collect()
    assert reference() is None


def test_projected_and_shifted_diagnostics_are_not_exact_unshifted():
    states = []
    def policy(state):
        if state.newton_correction_norm is None:
            return False
        states.append(state)
        return True
    solver.newton_solve(Quadratic(), dof.DofReduction(2, []), np.ones(2),
                         solver.NewtonOptions(), convergence=policy)
    assert states[-1].project_hessian and states[-1].diagonal_shifts == 0

    class Concave(Quadratic):
        def eval_values(self, x, need_gradient, need_hessian):
            return (-.5*x@x, -x if need_gradient else None,
                    -sp.eye(2, format='csc') if need_hessian else None)
    solve(Concave(), policy)
    assert not states[-1].project_hessian and states[-1].diagonal_shifts > 0
    assert states[-1].linear_backward_error < 1e-12
