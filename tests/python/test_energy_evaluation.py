"""Lazy scalar evaluations, borrowing bindings, and PSD projection contracts."""
import gc
import weakref

import numpy as np
import pytest
import scipy.sparse as sp
from litepgo import energy


class Counted(energy.FixedPatternEnergy):
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        self.calls = []
        super().__init__(sp.csc_matrix(self.matrix))

    def eval_values(self, x, need_gradient, need_hessian):
        self.calls.append((need_gradient, need_hessian))
        return (.5 * x @ self.matrix @ x,
                self.matrix @ x if need_gradient else None,
                self.matrix if need_hessian else None)


def test_lazy_input_copy_joint_cache_and_result_ownership():
    model = Counted(np.eye(2))
    x = np.array([1., 2.])
    e = model.evaluate(x)
    x[:] = 9
    assert model.calls == []
    H = e.hessian()
    g = e.gradient()
    assert e.value == 2.5
    assert model.calls == [(True, True)]
    H.data[:] = 100
    g[:] = 100
    np.testing.assert_array_equal(e.hessian().toarray(), np.eye(2))
    np.testing.assert_array_equal(e.gradient(), [1., 2.])
    np.testing.assert_array_equal(e.projected_hessian().toarray(), np.eye(2))
    assert model.calls == [(True, True)]
    assert e.num_dofs == 2
    for name in ('eval', 'eval_value', 'eval_gradient', 'eval_hessian', 'max_feasible_step'):
        assert not hasattr(model, name)
    assert not hasattr(energy, 'HessianKind')


def test_value_only_and_interleaved_evaluations():
    model = Counted(np.eye(2))
    a, b = model.evaluate(np.ones(2)), model.evaluate(np.full(2, 2.))
    assert a.value == 1.
    assert model.calls == [(False, False)]
    assert b.value == 4.
    np.testing.assert_array_equal(a.gradient(), [1., 1.])
    np.testing.assert_array_equal(b.gradient(), [2., 2.])
    assert a.value == 1.


@pytest.mark.parametrize('wrap', [lambda m: m,
    lambda m: energy.WeightedSumEnergy([(m, 2.)]),
    lambda m: energy.AffineMappedEnergy(m, np.eye(2))])
def test_binding_keeps_borrowed_model_and_children_alive(wrap):
    leaf = Counted(np.eye(2))
    model = wrap(leaf)
    ref = weakref.ref(leaf)
    e = model.evaluate(np.ones(2))
    del model, leaf
    gc.collect()
    assert ref() is not None
    e.hessian()
    e.gradient()
    e.projected_hessian()
    assert ref().calls == [(True, True)]
    del e
    gc.collect()
    assert ref() is None


def test_default_projection_can_fill_exact_pattern():
    H = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    model = Counted(H)
    e = model.evaluate(np.ones(3))
    P = e.projected_hessian().toarray()
    vals, vecs = np.linalg.eigh(H)
    np.testing.assert_allclose(P, (vecs * np.maximum(vals, 0)) @ vecs.T, atol=1e-14)
    assert P[0, 2] != 0
    np.testing.assert_array_equal(e.hessian().toarray(), H)
    assert model.calls == [(True, True)]


def test_projected_override_does_not_overwrite_exact_cache():
    class Fast(Counted):
        def eval_projected_values(self, x, need_gradient, need_hessian):
            self.calls.append(('projected', need_gradient, need_hessian))
            return (.5 * x @ self.matrix @ x, self.matrix @ x, np.eye(2))
    model = Fast(np.diag([-2., 3.]))
    e = model.evaluate(np.ones(2))
    np.testing.assert_array_equal(e.projected_hessian().toarray(), np.eye(2))
    np.testing.assert_array_equal(e.gradient(), [-2., 3.])
    assert model.calls == [('projected', True, True)]
    np.testing.assert_array_equal(e.hessian().toarray(), np.diag([-2., 3.]))
    e.projected_hessian()
    assert model.calls == [('projected', True, True), (True, True)]


def test_negative_weight_projection_uses_exact_sum():
    a = energy.QuadraticEnergy(np.diag([-2., 3.]))
    b = energy.QuadraticEnergy(np.eye(2))
    model = energy.WeightedSumEnergy([(a, 1.), (b, -1.)])
    e = model.evaluate(np.ones(2))
    np.testing.assert_array_equal(e.hessian().toarray(), np.diag([-3., 2.]))
    np.testing.assert_array_equal(e.projected_hessian().toarray(), np.diag([0., 2.]))


def test_feasible_step_checks_direction_and_default():
    model = Counted(np.eye(2))
    e = model.evaluate(np.ones(2))
    assert e.max_feasible_step(np.ones(2)) == 1.
    assert model.calls == []
    for dx in (np.zeros(1), np.full(2, np.nan)):
        with pytest.raises(ValueError):
            e.max_feasible_step(dx)


@pytest.mark.parametrize('mapped', [False, True])
def test_composites_allow_projection_fill_and_rebuild_plans(mapped):
    H = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    leaf = Counted(H)
    T = np.array([[1., 0.], [0., 1.], [1., 2.]])
    model = energy.AffineMappedEnergy(leaf, T) if mapped else energy.WeightedSumEnergy([(leaf, 2.)])
    vals, vecs = np.linalg.eigh(H)
    P = (vecs * np.maximum(vals, 0)) @ vecs.T
    expected_H, expected_P = (T.T @ H @ T, T.T @ P @ T) if mapped else (2 * H, 2 * P)
    for projected_first in (True, False, True):
        e = model.evaluate(np.ones(model.num_dofs))
        if projected_first:
            np.testing.assert_allclose(e.projected_hessian().toarray(), expected_P, atol=1e-14)
        np.testing.assert_allclose(e.hessian().toarray(), expected_H, atol=1e-14)
        np.testing.assert_allclose(e.projected_hessian().toarray(), expected_P, atol=1e-14)
