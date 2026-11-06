import numpy as np
import pytest
import scipy.sparse as sp

from litepgo import energy, solver


class Piecewise(energy.DynamicPatternResponse):
    def __init__(self):
        super().__init__(sp.eye(2, format="csc"))
        self.inputs = []

    def jacobian_pattern_at(self, x):
        self.inputs.append((x, x.copy()))
        return np.array([[1., 0.], [float(x[0] > 0), 1.]])

    def eval_impl(self, x, need_jacobian):
        positive = max(0., x[0])
        r = np.array([x[0] - 1, x[1] + positive**2 - 2])
        return (r, np.array([[1., 0.], [2*positive, 1.]])) if need_jacobian else r


def test_pattern_growth_shrinkage_and_newton():
    r = Piecewise()
    assert not r.has_fixed_pattern
    np.testing.assert_array_equal(r.eval_value(np.ones(2)), [0., 0.])
    assert r.eval_jacobian(np.ones(2)).nnz == 3
    assert not r.update_patterns(np.ones(2))
    assert not r.update_patterns(np.array([2., 1.]))
    assert r.eval_jacobian(np.ones(2)).nnz == 3
    # Ordinary evaluation automatically shrinks the structure at the new point.
    J = r.eval_jacobian(-np.ones(2))
    assert J.nnz == 2 and J[1, 0] == 0
    assert not r.update_patterns(-np.ones(2))
    assert r.jacobian_pattern.nnz == 2
    result = solver.gauss_newton_solve(r, -np.ones(2), solver.GaussNewtonOptions(gradient_tolerance=1e-9))
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, [1., 1.], atol=1e-8)
    assert r.jacobian_pattern.nnz == 3
    for x, saved in r.inputs:
        np.testing.assert_array_equal(x, saved)


def test_required_overrides_and_invalid_pattern():
    base = energy.DynamicPatternResponse(sp.eye(2))
    with pytest.raises(RuntimeError, match="jacobian_pattern_at"):
        base.update_patterns(np.ones(2))
    with pytest.raises(RuntimeError, match="eval_impl"):
        base.eval_value(np.ones(2))

    class WrongShape(Piecewise):
        def jacobian_pattern_at(self, x):
            return sp.eye(3)
    r = WrongShape()
    with pytest.raises(RuntimeError, match="dimensions changed"):
        r.update_patterns(np.ones(2))
    assert r.num_inputs == r.num_outputs == 2 and r.jacobian_pattern.nnz == 2

    class Broken(Piecewise):
        def jacobian_pattern_at(self, x):
            raise KeyError("pattern bug")
    with pytest.raises(KeyError, match="pattern bug"):
        Broken().update_patterns(np.ones(2))


@pytest.mark.parametrize("kind", ["response", "energy"])
def test_sparse_column_merge_retains_zero_slots_and_rejects_outside_entries(kind):
    shape = (7, 1) if kind == "response" else (7, 7)
    pattern = sp.csc_matrix(([1., 1., 1.], ([1, 3, 5], [0, 0, 0])), shape=shape)
    outside = 0.

    def derivative():
        return sp.csc_matrix(([0., 2., outside, 4., 0.],
                              ([0, 1, 2, 3, 6], [0, 0, 0, 0, 0])), shape=shape)

    class SparseResponse(energy.FixedPatternResponse):
        def eval_impl(self, x, need):
            return (np.zeros(7), derivative()) if need else np.zeros(7)

    class SparseEnergy(energy.FixedPatternEnergy):
        def eval_values(self, x, want_gradient, hessian_kind):
            return 0., np.zeros(7) if want_gradient else None, derivative() if hessian_kind is not None else None

    obj = SparseResponse(pattern) if kind == "response" else SparseEnergy(7, pattern)
    evaluate = obj.eval_jacobian if kind == "response" else obj.eval_hessian
    x = np.zeros(shape[1])
    actual = evaluate(x)
    np.testing.assert_array_equal(actual.indices, [1, 3, 5])
    np.testing.assert_array_equal(actual.indptr, pattern.indptr)
    np.testing.assert_array_equal(actual.data, [2., 4., 0.])
    outside = 1.
    with pytest.raises(RuntimeError, match="outside pattern"):
        evaluate(x)
    outside = 0.
    np.testing.assert_array_equal(evaluate(x).data, [2., 4., 0.])


def test_products_prepare_dynamic_jacobian_without_explicit_update():
    r = Piecewise()
    for sign in [1., -1., 1.]:
        x = np.array([sign, .4])
        J = np.array([[1., 0.], [2 * max(0., sign), 1.]])
        np.testing.assert_allclose(r.eval_jvp(x, np.ones(2)), J @ np.ones(2))
        np.testing.assert_allclose(r.eval_vjp(x, weights=np.ones(2)), J.T @ np.ones(2))
        assert r.jacobian_pattern.nnz == (3 if sign > 0 else 2)


def test_value_and_direct_products_do_not_prepare_patterns():
    class Direct(Piecewise):
        def jacobian_pattern_at(self, x):
            raise AssertionError("unneeded pattern preparation")

        def eval_jvp_impl(self, x, dx):
            return np.array([dx[0], 2 * max(0., x[0]) * dx[0] + dx[1]])

        def eval_vjp_impl(self, x, weights):
            return np.array([weights[0] + 2 * max(0., x[0]) * weights[1], weights[1]])

    r = Direct()
    x = np.ones(2)
    np.testing.assert_allclose(r.eval_value(x), [0, 0])
    np.testing.assert_allclose(r.eval_jvp(x, x), [1, 3])
    np.testing.assert_allclose(r.eval_vjp(x, x), [3, 1])
    assert r.jacobian_pattern.nnz == 2


def test_pattern_hook_reentry_is_rejected_and_guard_recovers():
    class Reentrant(Piecewise):
        reenter = True

        def jacobian_pattern_at(self, x):
            if self.reenter:
                self.eval_value(x)
            return super().jacobian_pattern_at(x)

    r = Reentrant()
    with pytest.raises(RuntimeError):
        r.eval_jacobian(np.ones(2))
    r.reenter = False
    value, J = r.eval_value_and_jacobian(np.ones(2))
    np.testing.assert_allclose(value, [0, 0])
    np.testing.assert_allclose(J.toarray(), [[1, 0], [2, 1]])
