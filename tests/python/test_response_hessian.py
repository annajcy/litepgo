import gc
import weakref

import numpy as np
import pytest
from litepgo import energy


class Affine(energy.FixedPatternResponse):
    def __init__(self, A, second=True):
        self.A = np.asarray(A, dtype=float)
        n = self.A.shape[1]
        super().__init__(self.A, hessian_pattern=np.zeros((n, n)) if second else None)

    def eval_impl(self, x, need_jacobian):
        y = self.A @ x + 1
        return (y, self.A) if need_jacobian else y

    def eval_hessian_impl(self, x, weights):
        return np.zeros((self.num_inputs, self.num_inputs))


def check_hessian(r, x, w):
    r.update_patterns(x)
    H = r.eval_hessian(x, weights=w).toarray()
    columns = []
    for d in np.eye(len(x)) * 1e-6:
        r.update_patterns(x + d)
        plus = r.eval_vjp(x + d, w)
        r.update_patterns(x - d)
        columns.append((plus - r.eval_vjp(x - d, w)) / 2e-6)
    np.testing.assert_allclose(H, np.column_stack(columns), atol=1e-8, rtol=1e-7)
    r.update_patterns(x)
    return H


def test_affine_support_is_explicit_and_old_subclasses_work():
    A = [[0.5, 0.2, 0.3], [-0.2, 1, 0.7]]
    r, old = Affine(A), Affine(A, second=False)
    x, w = np.array([-.2, .3, .1]), np.array([.2, -.7])
    assert r.has_hessian and not old.has_hessian
    H = r.eval_hessian(x, w)
    assert H.shape == (3, 3) and H.nnz == 0
    for action in [lambda: old.hessian_pattern, lambda: old.eval_hessian(x, w)]:
        with pytest.raises(RuntimeError, match="support"):
            action()
    np.testing.assert_array_equal(r.eval_value(x), old.eval_value(x))
    np.testing.assert_array_equal(r.eval_jacobian(x).toarray(), old.eval_jacobian(x).toarray())
    check_hessian(r, x, w)
    composite = energy.ComposedResponse(energy.ExpResponse(2), old)
    assert not composite.has_hessian
    with pytest.raises(RuntimeError, match="support"):
        composite.eval_hessian(x, w)


def test_nonlinear_composition_and_weight_independent_pattern():
    exp = energy.ExpResponse(2)
    r = energy.ComposedResponse(exp, exp)
    x, w = np.array([-.3, .2]), np.array([1.3, -.7])
    H = check_hessian(r, x, w)
    e = np.exp(x)
    np.testing.assert_allclose(H, np.diag(w * np.exp(e) * (e**2 + e)))
    zero = r.eval_hessian(x, np.zeros(2))
    assert zero.nnz == 2
    np.testing.assert_array_equal(zero.data, 0)
    np.testing.assert_array_equal(zero.indices, r.hessian_pattern.indices)
    np.testing.assert_array_equal(r.hessian_pattern.data, 0)
    # Exposed patterns are independent copies.
    pattern = r.hessian_pattern
    pattern.data[:] = 99
    np.testing.assert_array_equal(r.hessian_pattern.data, 0)
    sample = Affine([[.5, .2, .3]])
    nonlinear = energy.ComposedResponse(energy.ExpResponse(1), sample)
    c = np.array([-.2, .1, .4])
    actual = check_hessian(nonlinear, c, np.array([2.]))
    np.testing.assert_allclose(actual, 2 * np.exp((sample.A @ c + 1)[0]) * (sample.A.T @ sample.A))
    check_hessian(energy.ComposedResponse(Affine([[2., -3.]]), exp), x, np.array([.7]))


class Flip(energy.DynamicPatternResponse):
    def __init__(self):
        super().__init__(np.ones((2, 2)), initial_hessian_pattern=np.zeros((2, 2)))

    def jacobian_pattern_at(self, x):
        return np.ones((2, 2))

    def hessian_pattern_at(self, x):
        return np.diag([float(x[0] > 0), 0.])

    def eval_impl(self, x, need_jacobian):
        s = max(x[0], 0.)
        y = np.array([-x[0], x[1] + s**3])
        return (y, np.array([[-1., 0.], [3 * s**2, 1.]])) if need_jacobian else y

    def eval_hessian_impl(self, x, weights):
        return np.diag([6 * max(x[0], 0.) * weights[1], 0.])


def test_dynamic_hessian_changes_without_jacobian_change_and_shared_children():
    child = Flip()
    x, w = np.array([.3, .4]), np.array([.2, -.7])
    J = child.jacobian_pattern
    assert child.update_patterns(x)
    np.testing.assert_array_equal(J.indices, child.jacobian_pattern.indices)
    assert not child.update_patterns(x)
    assert child.update_patterns(-x)
    np.testing.assert_allclose(child.eval_hessian(x, w).toarray(), np.diag([6 * x[0] * w[1], 0.]))
    r = energy.ComposedResponse(child, child)
    np.testing.assert_allclose(r.eval_hessian(x, w).toarray(), np.diag([6 * x[0] * w[1], 0.]))
    for sign in [1, -1]:
        point = sign * x
        r.update_patterns(point)
        child.update_patterns(-point)
        np.testing.assert_allclose(r.eval_hessian(point, w).toarray(), np.diag([6 * abs(point[0]) * w[1], 0.]))
        check_hessian(r, point, w)
    assert child.update_patterns(np.zeros(2)) or child.hessian_pattern.nnz == 0
    np.testing.assert_array_equal(child.eval_hessian(np.zeros(2), w).toarray(), 0)


@pytest.mark.parametrize("x,w", [([0.], [1., 1.]), ([0., 0.], [1.]), ([np.nan, 0.], [1., 1.]), ([0., 0.], [np.inf, 1.])])
def test_invalid_inputs(x, w):
    with pytest.raises(ValueError):
        energy.ExpResponse(2).eval_hessian(np.array(x), np.array(w))


def test_python_hessian_hook_validation_and_exception_recovery():
    class Custom(energy.FixedPatternResponse):
        def __init__(self, pattern=np.ones((2, 2))):
            super().__init__(np.eye(2), hessian_pattern=pattern)
            self.result = np.eye(2)

        def eval_impl(self, x, need_jacobian):
            return (x, np.eye(2)) if need_jacobian else x

        def eval_hessian_impl(self, x, weights):
            # The arguments are owning copies, like existing value/Jacobian hooks.
            x[:] = 10
            weights[:] = 20
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    for invalid in [np.ones((1, 1)), np.array([[1., 1.], [0., 1.]])]:
        with pytest.raises(ValueError):
            Custom(invalid)
    r = Custom()
    x, w = np.zeros(2), np.ones(2)
    for invalid in [np.zeros((1, 1)), np.full((2, 2), np.nan), np.array([[1., 2.], [0., 1.]])]:
        r.result = invalid
        with pytest.raises(RuntimeError, match="Hessian"):
            r.eval_hessian(x, w)
    r.result = None
    with pytest.raises(ValueError, match="infeasible"):
        r.eval_hessian(x, w)
    r.result = ZeroDivisionError("custom failure")
    with pytest.raises(ZeroDivisionError, match="custom failure"):
        r.eval_hessian(x, w)
    r.result = np.eye(2)
    np.testing.assert_array_equal(r.eval_hessian(x, w).toarray(), np.eye(2))
    np.testing.assert_array_equal(x, 0)
    np.testing.assert_array_equal(w, 1)
    r = Custom(np.eye(2))
    r.result = np.ones((2, 2))
    with pytest.raises(RuntimeError, match="outside pattern"):
        r.eval_hessian(x, w)


def test_missing_hook_reentrancy_and_ownership():
    class Missing(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.eye(2), hessian_pattern=np.eye(2))

    with pytest.raises(RuntimeError):
        Missing().eval_hessian(np.zeros(2), np.ones(2))

    class Recursive(Missing):
        def eval_hessian_impl(self, x, weights):
            return self.eval_hessian(x, weights)

    with pytest.raises(RuntimeError, match="already in use"):
        Recursive().eval_hessian(np.zeros(2), np.ones(2))
    child = Affine(np.eye(2))
    owner = energy.ComposedResponse(energy.ExpResponse(2), child)
    child.owner = owner
    ref = weakref.ref(child)
    del child
    gc.collect()
    owner.eval_hessian(np.zeros(2), np.ones(2))
    assert ref() is not None
    del owner
    gc.collect()
    assert ref() is None


def test_dynamic_constant_hessian_and_atomic_pattern_validation():
    class Dynamic(energy.DynamicPatternResponse):
        def __init__(self):
            super().__init__(np.eye(2), initial_hessian_pattern=np.eye(2))

        def jacobian_pattern_at(self, x):
            return np.ones((2, 2))

        def eval_hessian_impl(self, x, weights):
            return np.diag(2 * weights)

    r = Dynamic()
    assert r.update_patterns(np.zeros(2))
    np.testing.assert_array_equal(r.eval_hessian(np.zeros(2), np.ones(2)).toarray(), 2 * np.eye(2))

    class Invalid(Dynamic):
        def hessian_pattern_at(self, x):
            return np.ones((3, 3))

    r = Invalid()
    before = r.jacobian_pattern
    with pytest.raises(ValueError, match="Hessian dimensions"):
        r.update_patterns(np.zeros(2))
    np.testing.assert_array_equal(r.jacobian_pattern.indptr, before.indptr)
    np.testing.assert_array_equal(r.jacobian_pattern.indices, before.indices)
