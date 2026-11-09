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
    H = r.evaluate(x).hessian(weights=w).toarray()
    columns = []
    for d in np.eye(len(x)) * 1e-6:
        plus = r.evaluate(x + d).vjp(w)
        columns.append((plus - r.evaluate(x - d).vjp(w)) / 2e-06)
    np.testing.assert_allclose(H, np.column_stack(columns), atol=1e-8, rtol=1e-7)
    return H


def test_affine_support_is_explicit_and_old_subclasses_work():
    A = [[0.5, 0.2, 0.3], [-0.2, 1, 0.7]]
    r, old = Affine(A), Affine(A, second=False)
    x, w = np.array([-.2, .3, .1]), np.array([.2, -.7])
    assert r.has_hessian and not old.has_hessian
    H = r.evaluate(x).hessian(w)
    assert H.shape == (3, 3) and H.nnz == 0
    for action in [lambda: old.evaluate(x).hessian(w)]:
        with pytest.raises(RuntimeError, match="support"):
            action()
    np.testing.assert_array_equal(r.evaluate(x).value, old.evaluate(x).value)
    np.testing.assert_array_equal(r.evaluate(x).jacobian().toarray(), old.evaluate(x).jacobian().toarray())
    check_hessian(r, x, w)
    composite = energy.ComposedResponse(energy.ExpResponse(2), old)
    assert not composite.has_hessian
    with pytest.raises(RuntimeError, match="support"):
        composite.evaluate(x).hessian(w)


def test_nonlinear_composition_and_weight_independent_pattern():
    exp = energy.ExpResponse(2)
    r = energy.ComposedResponse(exp, exp)
    x, w = np.array([-.3, .2]), np.array([1.3, -.7])
    H = check_hessian(r, x, w)
    e = np.exp(x)
    np.testing.assert_allclose(H, np.diag(w * np.exp(e) * (e**2 + e)))
    zero = r.evaluate(x).hessian(np.zeros(2))
    assert zero.nnz == 2
    np.testing.assert_array_equal(zero.data, 0)
    saved = r.evaluate(x).hessian(w)
    saved.data[:] = 99
    np.testing.assert_allclose(r.evaluate(x).hessian(w).toarray(), H)
    sample = Affine([[.5, .2, .3]])
    nonlinear = energy.ComposedResponse(energy.ExpResponse(1), sample)
    c = np.array([-.2, .1, .4])
    actual = check_hessian(nonlinear, c, np.array([2.]))
    np.testing.assert_allclose(actual, 2 * np.exp((sample.A @ c + 1)[0]) * (sample.A.T @ sample.A))
    check_hessian(energy.ComposedResponse(Affine([[2., -3.]]), exp), x, np.array([.7]))


class Flip(energy.DynamicPatternResponse):
    @property
    def has_hessian(self):
        return True

    def __init__(self):
        super().__init__(2, 2)

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
    np.testing.assert_allclose(child.evaluate(x).hessian(w).toarray(), np.diag([6 * x[0] * w[1], 0.0]))
    r = energy.ComposedResponse(child, child)
    np.testing.assert_allclose(r.evaluate(x).hessian(w).toarray(), np.diag([6 * x[0] * w[1], 0.0]))
    for sign in [1, -1]:
        point = sign * x
        np.testing.assert_allclose(r.evaluate(point).hessian(w).toarray(), np.diag([6 * abs(point[0]) * w[1], 0.0]))
        check_hessian(r, point, w)
    np.testing.assert_array_equal(child.evaluate(np.zeros(2)).hessian(w).toarray(), 0)


@pytest.mark.parametrize("x,w", [([0.], [1., 1.]), ([0., 0.], [1.]), ([np.nan, 0.], [1., 1.]), ([0., 0.], [np.inf, 1.])])
def test_invalid_inputs(x, w):
    with pytest.raises(ValueError):
        energy.ExpResponse(2).evaluate(np.array(x)).hessian(np.array(w))


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
    for invalid in [np.zeros((1, 1)), np.full((2, 2), np.nan)]:
        r.result = invalid
        with pytest.raises(RuntimeError, match="Hessian"):
            r.evaluate(x).hessian(w)
    r.result = None
    with pytest.raises(ValueError, match="infeasible"):
        r.evaluate(x).hessian(w)
    r.result = ZeroDivisionError("custom failure")
    with pytest.raises(ZeroDivisionError, match="custom failure"):
        r.evaluate(x).hessian(w)
    r.result = np.eye(2)
    np.testing.assert_array_equal(r.evaluate(x).hessian(w).toarray(), np.eye(2))
    np.testing.assert_array_equal(x, 0)
    np.testing.assert_array_equal(w, 1)
    r = Custom(np.eye(2))
    r.result = np.ones((2, 2))
    with pytest.raises(RuntimeError, match="outside pattern"):
        r.evaluate(x).hessian(w)


def test_missing_hook_reentrancy_and_ownership():
    class Missing(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.eye(2), hessian_pattern=np.eye(2))

    with pytest.raises(RuntimeError):
        Missing().evaluate(np.zeros(2)).hessian(np.ones(2))

    child = Affine(np.eye(2))
    owner = energy.ComposedResponse(energy.ExpResponse(2), child)
    child.owner = weakref.ref(owner)
    assert child.owner() is owner
    owner_ref = weakref.ref(owner)
    ref = weakref.ref(child)
    del child
    gc.collect()
    owner.evaluate(np.zeros(2)).hessian(np.ones(2))
    assert ref() is not None
    del owner
    gc.collect()
    assert ref() is None
    assert owner_ref() is None


def test_dynamic_constant_hessian_and_atomic_pattern_validation():
    class Dynamic(energy.DynamicPatternResponse):
        @property
        def has_hessian(self):
            return True

        def hessian_pattern_at(self, x):
            return np.eye(2)

        def __init__(self):
            super().__init__(2, 2)

        def jacobian_pattern_at(self, x):
            return np.ones((2, 2))

        def eval_hessian_impl(self, x, weights):
            return np.diag(2 * weights)

    r = Dynamic()
    np.testing.assert_array_equal(r.evaluate(np.zeros(2)).hessian(np.ones(2)).toarray(), 2 * np.eye(2))

    class Invalid(Dynamic):
        def hessian_pattern_at(self, x):
            return np.ones((3, 3))

    r = Invalid()
    with pytest.raises(ValueError, match="pattern dimensions"):
        r.evaluate(np.zeros(2)).hessian(np.ones(2))
