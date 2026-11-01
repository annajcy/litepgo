import numpy as np
import pytest
import scipy.sparse as sp

from litepgo import response, solver


class Piecewise(response.DynamicPatternResponse):
    def __init__(self):
        super().__init__(sp.eye(2, format="csc"))
        self.inputs = []

    def pattern_at(self, x):
        self.inputs.append((x, x.copy()))
        return np.array([[1., 0.], [float(x[0] > 0), 1.]])

    def eval_values(self, x, need_jacobian):
        positive = max(0., x[0])
        r = np.array([x[0] - 1, x[1] + positive**2 - 2])
        return (r, np.array([[1., 0.], [2*positive, 1.]])) if need_jacobian else r


def test_pattern_growth_shrinkage_and_newton():
    r = Piecewise()
    assert not r.has_fixed_pattern
    np.testing.assert_array_equal(r.eval_value(np.ones(2)), [0., 0.])
    with pytest.raises(RuntimeError, match="outside pattern"):
        r.eval_jacobian(np.ones(2))
    assert r.update_pattern(np.ones(2))
    assert not r.update_pattern(np.array([2., 1.]))
    assert r.eval_jacobian(np.ones(2)).nnz == 3
    # A stale superset is valid, but unused slots must be zeroed.
    J = r.eval_jacobian(-np.ones(2))
    assert J.nnz == 3 and J[1, 0] == 0
    assert r.update_pattern(-np.ones(2))
    assert r.jacobian_pattern.nnz == 2
    result = solver.gauss_newton_solve(r, -np.ones(2), solver.GaussNewtonOptions(gradient_tolerance=1e-9))
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, [1., 1.], atol=1e-8)
    assert r.jacobian_pattern.nnz == 3
    for x, saved in r.inputs:
        np.testing.assert_array_equal(x, saved)


def test_required_overrides_and_invalid_pattern():
    base = response.DynamicPatternResponse(sp.eye(2))
    with pytest.raises(RuntimeError, match="pattern_at"):
        base.update_pattern(np.ones(2))
    with pytest.raises(RuntimeError, match="eval_values"):
        base.eval_value(np.ones(2))

    class WrongShape(Piecewise):
        def pattern_at(self, x):
            return sp.eye(3)
    r = WrongShape()
    with pytest.raises(RuntimeError, match="dimensions changed"):
        r.update_pattern(np.ones(2))
    assert r.num_inputs == r.num_outputs == 2 and r.jacobian_pattern.nnz == 2

    class Broken(Piecewise):
        def pattern_at(self, x):
            raise KeyError("pattern bug")
    with pytest.raises(KeyError, match="pattern bug"):
        Broken().update_pattern(np.ones(2))


def test_sparse_column_merge_retains_zero_slots_and_rejects_outside_entries():
    class SparseResponse(response.FixedPatternResponse):
        def __init__(self):
            super().__init__(sp.csc_matrix(([1., 1., 1.], ([1, 3, 5], [0, 0, 0])), shape=(7, 1)))
            self.outside = 0.

        def eval_values(self, x, need):
            J = sp.csc_matrix(([0., 2., self.outside, 4., 0.],
                               ([0, 1, 2, 3, 6], [0, 0, 0, 0, 0])), shape=(7, 1))
            return (np.zeros(7), J) if need else np.zeros(7)

    r = SparseResponse()
    J = r.eval_jacobian(np.zeros(1))
    np.testing.assert_array_equal(J.indices, [1, 3, 5])
    np.testing.assert_array_equal(J.data, [2., 4., 0.])
    r.outside = 1.
    with pytest.raises(RuntimeError, match="outside pattern"):
        r.eval_jacobian(np.zeros(1))
