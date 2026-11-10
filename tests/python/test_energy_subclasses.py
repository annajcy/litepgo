import gc
import weakref
import numpy as np
import pytest
import scipy.sparse as sp

from litepgo import dof, energy, solver


class Quadratic(energy.FixedPatternEnergy):
    def __init__(self):
        super().__init__(sp.eye(2, format="csc"))
        self.calls = []
        self.steps = 0

    def eval_values(self, x, want_gradient, need_hessian):
        self.calls.append((x, x.copy(), want_gradient, need_hessian))
        r = x - [1., 2.]
        return .5 * r @ r, r if want_gradient else None, np.eye(2) if need_hessian else None

    def feasible_step(self, x, dx):
        self.steps += 1
        return 1.


class Changing(energy.DynamicPatternEnergy):
    def __init__(self):
        super().__init__(2)
        self.pattern_calls = 0

    def hessian_pattern_at(self, x):
        self.pattern_calls += 1
        return np.ones((2, 2)) if x.sum() > 0 else sp.eye(2)

    def eval_values(self, x, want_gradient, need_hessian):
        s = max(x.sum(), 0.)
        r = x - 1
        return (.5 * r @ r + .25 * s**4,
                r + s**3 if want_gradient else None,
                np.eye(2) + 3*s*s*np.ones((2, 2)) if need_hessian else None)


def test_fixed_subclass_derivatives_newton_and_lifetime():
    q = Quadratic()
    assert q.num_dofs == 2
    assert q.evaluate(np.zeros(2)).value == 2.5
    np.testing.assert_array_equal(q.evaluate(np.zeros(2)).gradient(), [-1., -2.])
    np.testing.assert_array_equal(q.evaluate(np.zeros(2)).hessian().toarray(), np.eye(2))
    q.evaluate(np.zeros(2)).projected_hessian()
    assert q.calls[-1][3] == True
    result = solver.newton_solve(q, dof.DofReduction(2, []), np.zeros(2),
                                 solver.NewtonOptions(gradient_tolerance=1e-10))
    assert result.status == solver.NewtonStatus.Converged and q.steps > 0
    np.testing.assert_allclose(result.x, [1., 2.], atol=1e-10)
    for x, snapshot, _, _ in q.calls:
        np.testing.assert_array_equal(x, snapshot)
    composite = energy.WeightedSumEnergy([(q, 2.)])
    del q
    gc.collect()
    assert composite.evaluate(np.zeros(2)).value == 5.


def test_dynamic_pattern_and_newton():
    q = Changing()

    assert q.evaluate(np.ones(2)).value == 4.
    assert q.pattern_calls == 0
    active = q.evaluate(np.ones(2)).hessian()
    assert active.nnz == 4


    assert q.evaluate(np.ones(2)).hessian().nnz == 4

    assert q.evaluate(-np.ones(2)).hessian().nnz == 2
    result = solver.newton_solve(q, dof.DofReduction(2, []), -np.ones(2),
                                 solver.NewtonOptions(gradient_tolerance=1e-9))
    assert result.status == solver.NewtonStatus.Converged
    assert np.linalg.norm(q.evaluate(result.x).gradient(), np.inf) < 1e-9
    assert q.evaluate(result.x).hessian().nnz == 4 and q.pattern_calls > 3


def test_required_overrides_errors_and_feasibility():
    fixed = energy.FixedPatternEnergy(sp.eye(1))
    with pytest.raises(RuntimeError, match="eval_values"):
        fixed.evaluate(np.ones(1)).value
    dynamic = energy.DynamicPatternEnergy(1)
    with pytest.raises(RuntimeError, match="pattern_at"):
        dynamic.evaluate(np.ones(1)).hessian()
    class Infeasible(Quadratic):
        def eval_values(self, *args):
            return None
    with pytest.raises(ValueError, match="infeasible"):
        Infeasible().evaluate(np.zeros(2)).value

    class Broken(Quadratic):
        def eval_values(self, *args):
            raise KeyError("callback bug")
    with pytest.raises(KeyError, match="callback bug"):
        Broken().evaluate(np.zeros(2)).value

    class BadGradient(Quadratic):
        def eval_values(self, *args):
            return 0., np.zeros(3), None
    with pytest.raises(RuntimeError, match="invalid gradient"):
        BadGradient().evaluate(np.zeros(2)).gradient()

    class BadBound(Quadratic):
        def feasible_step(self, x, dx):
            return 2.
    with pytest.raises(RuntimeError):
        BadBound().evaluate(np.zeros(2)).max_feasible_step(np.ones(2))
    assert Changing().evaluate(np.zeros(2)).max_feasible_step(np.ones(2)) == 1.


def test_invalid_hessian_and_pattern_are_rejected():
    class BadHessian(Quadratic):
        def eval_values(self, x, want_gradient, need_hessian):
            return 0., np.zeros(2) if want_gradient else None, np.full((2, 2), np.nan)
    with pytest.raises(RuntimeError, match="invalid Hessian"):
        BadHessian().evaluate(np.zeros(2)).hessian()

    class BadPattern(Changing):
        def hessian_pattern_at(self, x):
            return sp.eye(3)
    with pytest.raises(ValueError, match="pattern"):
        BadPattern().evaluate(np.zeros(2)).hessian()

    class Plain(energy.FixedPatternEnergy):
        def __init__(self):
            super().__init__(sp.eye(1))
        def eval_values(self, x, want_gradient, need_hessian):
            return .5*x[0]**2, x if want_gradient else None, np.ones((1, 1)) if need_hessian else None
    assert Plain().evaluate(np.ones(1)).max_feasible_step(-np.ones(1)) == 1.


@pytest.mark.parametrize("kind", ["sum", "mapped", "nested", "repeated"])
def test_composite_lifetime_with_weak_back_reference(kind):
    child = Quadratic()
    if kind == "sum":
        owner = energy.WeightedSumEnergy([(child, 1.)])
    elif kind == "mapped":
        owner = energy.AffineMappedEnergy(child, sp.eye(2, format="csc"))
    elif kind == "nested":
        owner = energy.AffineMappedEnergy(
            energy.WeightedSumEnergy([(child, 1.)]), sp.eye(2, format="csc"))
    else:
        owner = energy.WeightedSumEnergy([(child, .5), (child, .5)])
    child.owner = weakref.ref(owner)
    assert child.owner() is owner
    owner_ref = weakref.ref(owner)
    ref = weakref.ref(child)
    del child
    gc.collect()
    assert ref() is not None
    assert owner.evaluate(np.zeros(2)).value == pytest.approx(2.5)
    del owner
    gc.collect()
    assert ref() is None
    assert owner_ref() is None
