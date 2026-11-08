import numpy as np
import pytest
from litepgo import energy


class Dynamic(energy.DynamicPatternEnergy):
    def __init__(self):
        super().__init__(2)
        self.patterns = 0
        self.calls = 0
        self.reenter = False
        self.fail = False

    def hessian_pattern_at(self, x):
        self.patterns += 1
        return np.eye(2) if x.sum() <= 0 else np.ones((2, 2))

    def eval_values(self, x, gradient, kind):
        self.calls += 1
        if self.fail:
            raise KeyError("deliberate")
        if self.reenter:
            self.reenter = False
            self.eval_value(-x)
        s = max(0., x.sum())
        return (.5 * x @ x + .25 * s**4,
                x + s**3 if gradient else None,
                np.eye(2) + 3*s*s if kind is not None else None)


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("kind", [None, energy.HessianKind.Exact, energy.HessianKind.Projected])
def test_joint_requests_prepare_only_needed_structure(gradient, kind):
    e = Dynamic()
    value, g, H = e.eval(np.ones(2), gradient=gradient, hessian=kind)
    assert e.calls == 1 and e.patterns == int(kind is not None)
    assert value == 5
    assert (g is not None) == gradient
    assert (H is not None) == (kind is not None)


def test_dynamic_results_are_independent_and_hooks_can_reenter():
    e = Dynamic()
    e.reenter = True
    active = e.eval_hessian(np.ones(2))
    original = active.toarray().copy()
    assert e.eval_hessian(-np.ones(2)).nnz == 2
    np.testing.assert_array_equal(active.toarray(), original)
    np.testing.assert_array_equal(e.eval_hessian(np.ones(2)).toarray(), original)
    for name in ("update_pattern", "hessian_pattern", "has_fixed_pattern", "eval_projected_hessian"):
        assert not hasattr(e, name)


def test_shared_dynamic_children_and_cache_recovery():
    child = Dynamic()
    total = energy.WeightedSumEnergy([(child, 2), (child, -1)])
    mapped = energy.AffineMappedEnergy(total, np.array([[1., 2.], [0., 1.]]))
    for x in (np.ones(2), -np.ones(2), np.ones(2)):
        T = np.array([[1., 2.], [0., 1.]])
        expected = T.T @ child.eval_hessian(T @ x).toarray() @ T
        np.testing.assert_allclose(mapped.eval_hessian(x).toarray(), expected)
    child.fail = True
    with pytest.raises(KeyError, match="deliberate"):
        mapped.eval_hessian(np.ones(2))
    child.fail = False
    np.testing.assert_allclose(mapped.eval_hessian(np.ones(2)).toarray(), expected)


def test_projected_does_not_promise_psd():
    indefinite = energy.QuadraticEnergy(np.diag([1., -2.]))
    np.testing.assert_array_equal(indefinite.eval_hessian(np.zeros(2), kind=energy.HessianKind.Projected).toarray(),
                                  np.diag([1., -2.]))
    negative = energy.WeightedSumEnergy([(energy.QuadraticEnergy(np.eye(2)), -1.)])
    np.testing.assert_array_equal(negative.eval_hessian(np.zeros(2), kind=energy.HessianKind.Projected).toarray(),
                                  -np.eye(2))


def test_equal_nnz_different_structure_rebuilds_composite_plans():
    class MovingStructure(energy.DynamicPatternEnergy):
        def __init__(self):
            super().__init__(3)

        def hessian_pattern_at(self, x):
            p = np.eye(3)
            i = 1 if x[0] > 0 else 2
            p[0, i] = p[i, 0] = 1
            return p

        def eval_values(self, x, gradient, kind):
            # Both optional slots are structural zeros for this quadratic.
            return .5*x@x, x if gradient else None, np.eye(3) if kind is not None else None

    child = MovingStructure()
    total = energy.WeightedSumEnergy([(child, 2.)])
    T = np.array([[1., 2., 0.], [0., 1., 0.], [0., 0., 1.]])
    mapped = energy.AffineMappedEnergy(total, T)
    for sign in [1., -1., 1.]:
        x = np.array([sign, 0., 0.])
        H = total.eval_hessian(x)
        assert H.nnz == 5
        np.testing.assert_array_equal(H.indices, child.eval_hessian(x).indices)
        np.testing.assert_allclose(mapped.eval_hessian(x).toarray(), 2*T.T@T)


def test_fixed_pattern_snapshot_and_zero_dofs():
    pattern = np.eye(2)

    class Fixed(energy.FixedPatternEnergy):
        def eval_values(self, x, gradient, kind):
            return .5*x@x, x if gradient else None, np.eye(len(x)) if kind is not None else None

    e = Fixed(pattern)
    pattern[:] = 0
    assert e.eval_hessian(np.ones(2)).nnz == 2
    zero = Fixed(np.empty((0, 0)))
    value, g, H = zero.eval(np.empty(0), gradient=True, hessian=energy.HessianKind.Exact)
    assert value == 0 and g.shape == (0,) and H.shape == (0, 0)
