"""Execution contracts of state-free responses and workspace-owning mesh responses."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
from litepgo import dof, energy


def test_mesh_workspace_guard_covers_overlapping_calls_and_recovers():
    entered, release = Event(), Event()
    mapping = dof.StencilDofMap(1, [[0]], 1)

    class Blocking(energy.MeshResponse):
        block = True
        fail = False

        def eval_batch(self, ids, x, need):
            if self.block:
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("test worker timed out")
            if self.fail:
                raise ValueError("callback failed")
            return (x.copy(), np.ones_like(x)) if need else x.copy()

    r = Blocking(mapping, mapping)
    x = np.ones(1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(r.eval_value, x)
        try:
            assert entered.wait(10)
            for call in (lambda: r.eval_value(x), lambda: r.eval_jacobian(x),
                         lambda: r.eval_jvp(x, x), lambda: r.eval_vjp(x, x)):
                with pytest.raises(RuntimeError, match="already in use"):
                    call()
        finally:
            release.set()
        np.testing.assert_array_equal(future.result(timeout=10), x)
    r.block = False
    r.fail = True
    with pytest.raises(ValueError, match="callback failed"):
        r.eval_value(x)
    r.fail = False
    np.testing.assert_array_equal(r.eval_jvp(x, x), x)
    np.testing.assert_array_equal(r.eval_jacobian(x).toarray(), [[1]])


def test_mesh_workspace_guard_rejects_callback_reentry():
    mapping = dof.StencilDofMap(1, [[0]], 1)

    class Reentrant(energy.MeshResponse):
        reenter = True

        def eval_batch(self, ids, x, need):
            if self.reenter:
                self.eval_value(np.ones(1))
            return (x.copy(), np.ones_like(x)) if need else x.copy()

    r = Reentrant(mapping, mapping)
    with pytest.raises(RuntimeError, match="already in use"):
        r.eval_value(np.ones(1))
    r.reenter = False
    np.testing.assert_array_equal(r.eval_value(np.ones(1)), [1])


def test_dynamic_capability_errors_propagate_through_native_composition():
    class Dynamic(energy.DynamicPatternResponse):
        def __init__(self):
            super().__init__(1, 1)

        @property
        def has_hessian(self):
            raise ValueError("capability query failed")

        def jacobian_pattern_at(self, x):
            return np.eye(1)

        def eval_impl(self, x, need):
            return (x.copy(), np.eye(1)) if need else x.copy()

    r = energy.ComposedResponse(energy.ExpResponse(1), Dynamic())
    with pytest.raises(ValueError, match="capability query failed"):
        r.eval_hessian(np.ones(1), np.ones(1))
    np.testing.assert_allclose(r.eval_value(np.zeros(1)), [1])


def test_public_pattern_protocol_is_removed_and_dimensions_validated():
    r = energy.AffineMappedResponse(np.eye(2))
    for name in ("update_patterns", "has_fixed_pattern", "jacobian_pattern", "hessian_pattern",
                 "eval_jacobian_into", "eval_hessian_into", "eval_value_and_jacobian_into"):
        assert not hasattr(r, name)
    with pytest.raises(ValueError):
        energy.DynamicPatternResponse(-1, 2)


def test_composition_retains_structure_under_numerical_cancellation():
    duplicate = energy.AffineMappedResponse(np.ones((2, 1)))
    nonlinear = energy.ComposedResponse(energy.ExpResponse(2), duplicate)
    r = energy.ComposedResponse(energy.AffineMappedResponse(np.array([[1., -1.]])), nonlinear)
    x = np.array([.3])
    J = r.eval_jacobian(x)
    H = r.eval_hessian(x, np.ones(1))
    assert J.nnz == H.nnz == 1
    np.testing.assert_array_equal(J.data, 0)
    np.testing.assert_array_equal(H.data, 0)
    np.testing.assert_array_equal(r.eval_hessian(x, np.zeros(1)).indices, H.indices)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("operation", ["hessian", "jvp", "vjp"])
def test_derivative_hooks_allow_finite_reentry_and_recover(dynamic, operation):
    base = energy.DynamicPatternResponse if dynamic else energy.FixedPatternResponse

    class Reentrant(base):
        def __init__(self):
            if dynamic:
                super().__init__(1, 1)
            else:
                super().__init__(np.eye(1), hessian_pattern=np.eye(1))
            self.visited = []
            self.fail = False

        @property
        def has_hessian(self):
            return True

        def jacobian_pattern_at(self, x):
            return np.eye(1)

        def hessian_pattern_at(self, x):
            return np.eye(1)

        def eval_impl(self, x, need):
            return (x*x, np.diag(2*x)) if need else x*x

        def evaluate(self, kind, x, direction):
            self.visited.append((kind, float(x[0])))
            if x[0] > 0:
                result = getattr(self, "eval_" + kind)(-x, direction)
                return result if kind == "hessian" else -result
            if self.fail:
                raise ValueError("nested derivative failed")
            if kind == "hessian":
                return np.diag(2 * direction)
            # Calling super still explicitly selects the Jacobian fallback.
            return getattr(super(), "eval_" + kind + "_impl")(x, direction)

        def eval_hessian_impl(self, x, weights):
            return self.evaluate("hessian", x, weights)

        def eval_jvp_impl(self, x, dx):
            return self.evaluate("jvp", x, dx)

        def eval_vjp_impl(self, x, weights):
            return self.evaluate("vjp", x, weights)

    r = Reentrant()
    call = getattr(r, "eval_" + operation)
    x, direction = np.array([3.]), np.array([2.])
    actual = call(x, direction)
    np.testing.assert_allclose(actual.toarray() if operation == "hessian" else actual,
                               [[4.]] if operation == "hessian" else [12.])
    assert r.visited == [(operation, 3.), (operation, -3.)]
    r.fail = True
    with pytest.raises(ValueError, match="nested derivative failed"):
        call(x, direction)
    r.fail = False
    actual = call(x, direction)
    np.testing.assert_allclose(actual.toarray() if operation == "hessian" else actual,
                               [[4.]] if operation == "hessian" else [12.])
