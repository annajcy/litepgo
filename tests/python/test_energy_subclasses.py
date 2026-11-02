import gc
import numpy as np
import pytest
import scipy.sparse as sp

from litepgo import dof, energy, solver


class Quadratic(energy.FixedPatternEnergy):
    def __init__(self):
        super().__init__(2, sp.eye(2, format="csc"))
        self.calls = []
        self.steps = 0

    def eval_values(self, x, want_gradient, hessian_kind):
        self.calls.append((x, x.copy(), want_gradient, hessian_kind))
        r = x - [1., 2.]
        return .5 * r @ r, r if want_gradient else None, np.eye(2) if hessian_kind is not None else None

    def feasible_step(self, x, dx):
        self.steps += 1
        return 1.


class Changing(energy.DynamicPatternEnergy):
    def __init__(self):
        super().__init__(2, sp.eye(2, format="csc"))
        self.pattern_calls = 0

    def pattern_at(self, x):
        self.pattern_calls += 1
        return np.ones((2, 2)) if x.sum() > 0 else sp.eye(2)

    def eval_values(self, x, want_gradient, hessian_kind):
        s = max(x.sum(), 0.)
        r = x - 1
        return (.5 * r @ r + .25 * s**4,
                r + s**3 if want_gradient else None,
                np.eye(2) + 3*s*s*np.ones((2, 2)) if hessian_kind is not None else None)


def test_fixed_subclass_derivatives_newton_and_lifetime():
    q = Quadratic()
    assert q.has_fixed_pattern and not q.update_pattern(np.zeros(2))
    assert q.eval_value(np.zeros(2)) == 2.5
    np.testing.assert_array_equal(q.eval_gradient(np.zeros(2)), [-1., -2.])
    np.testing.assert_array_equal(q.eval_hessian(np.zeros(2)).toarray(), np.eye(2))
    q.eval_projected_hessian(np.zeros(2))
    assert q.calls[-1][3] == energy.HessianKind.Projected
    result = solver.newton_solve(q, dof.DofReduction(2, []), np.zeros(2),
                                 solver.NewtonOptions(gradient_tolerance=1e-10))
    assert result.status == solver.NewtonStatus.Converged and q.steps > 0
    np.testing.assert_allclose(result.x, [1., 2.], atol=1e-10)
    for x, snapshot, _, _ in q.calls:
        np.testing.assert_array_equal(x, snapshot)
    composite = energy.WeightedEnergySum([(q, 2.)])
    del q
    gc.collect()
    assert composite.eval_value(np.zeros(2)) == 5.


def test_dynamic_pattern_and_newton():
    q = Changing()
    assert not q.has_fixed_pattern
    assert not q.update_pattern(-np.ones(2))
    assert q.eval_value(np.ones(2)) == 4.
    with pytest.raises(RuntimeError, match="outside pattern"):
        q.eval_hessian(np.ones(2))
    assert q.update_pattern(np.ones(2))
    assert not q.update_pattern(np.ones(2))
    assert q.eval_hessian(np.ones(2)).nnz == 4
    assert q.update_pattern(-np.ones(2))
    assert q.eval_hessian(-np.ones(2)).nnz == 2
    result = solver.newton_solve(q, dof.DofReduction(2, []), -np.ones(2),
                                 solver.NewtonOptions(gradient_tolerance=1e-9))
    assert result.status == solver.NewtonStatus.Converged
    assert np.linalg.norm(q.eval_gradient(result.x), np.inf) < 1e-9
    assert q.hessian_pattern.nnz == 4 and q.pattern_calls > 3


def test_required_overrides_errors_and_feasibility():
    fixed = energy.FixedPatternEnergy(1, sp.eye(1))
    with pytest.raises(RuntimeError, match="eval_values"):
        fixed.eval_value(np.ones(1))
    dynamic = energy.DynamicPatternEnergy(1, sp.eye(1))
    with pytest.raises(RuntimeError, match="pattern_at"):
        dynamic.update_pattern(np.ones(1))

    class Infeasible(Quadratic):
        def eval_values(self, *args):
            return None
    with pytest.raises(ValueError, match="infeasible"):
        Infeasible().eval_value(np.zeros(2))

    class Broken(Quadratic):
        def eval_values(self, *args):
            raise KeyError("callback bug")
    with pytest.raises(KeyError, match="callback bug"):
        Broken().eval_value(np.zeros(2))

    class BadGradient(Quadratic):
        def eval_values(self, *args):
            return 0., np.zeros(3), None
    with pytest.raises(RuntimeError, match="invalid gradient"):
        BadGradient().eval_gradient(np.zeros(2))

    class BadBound(Quadratic):
        def feasible_step(self, x, dx):
            return 2.
    with pytest.raises(RuntimeError):
        BadBound().max_feasible_step(np.zeros(2), np.ones(2))
    assert Changing().max_feasible_step(np.zeros(2), np.ones(2)) == 1.


def test_invalid_hessian_and_pattern_are_rejected():
    class BadHessian(Quadratic):
        def eval_values(self, x, want_gradient, hessian_kind):
            return 0., None, np.full((2, 2), np.nan)
    with pytest.raises(RuntimeError, match="invalid Hessian"):
        BadHessian().eval_hessian(np.zeros(2))

    class BadPattern(Changing):
        def pattern_at(self, x):
            return sp.eye(3)
    with pytest.raises(ValueError, match="pattern"):
        BadPattern().update_pattern(np.zeros(2))

    class Plain(energy.FixedPatternEnergy):
        def __init__(self):
            super().__init__(1, sp.eye(1))
        def eval_values(self, x, want_gradient, hessian_kind):
            return .5*x[0]**2, x if want_gradient else None, np.ones((1, 1)) if hessian_kind is not None else None
    assert Plain().max_feasible_step(np.ones(1), -np.ones(1)) == 1.
