from equilibrium_fixtures import FormulaForce
import numpy as np
from scipy import sparse

from litepgo import dof, energy, solver


def test_direct_sensitivity_callbacks_and_fixed_values():
    calls = []

    def make_energy(p):
        return energy.QuadraticEnergy(sparse.eye(2, format="csc"), -p)

    def partial(p, u):
        calls.append(u.copy())
        return -sparse.eye(2, format="csc")

    response = solver.StaticEquilibriumResponse(
        2,
        dof.DofReduction(2, [0]),
        np.array([0.4, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        make_energy,
        lambda u: FormulaForce(2, u, make_energy, partial),
    )
    p = np.array([2.0, 3.0])
    u, J = response.eval_value_and_jacobian(p)
    np.testing.assert_allclose(u, [0.4, 3.0])
    np.testing.assert_allclose(J.toarray(), [[0, 0], [0, 1]])
    for c in range(2):
        delta = np.eye(2)[c] * 1e-4
        numerical = (
            response.eval_value(p + delta) - response.eval_value(p - delta)
        ) / 2e-4
        np.testing.assert_allclose(J.toarray()[:, c], numerical, atol=1e-10)
    assert len(calls) == 1


def test_bound_callbacks_are_kept_alive_and_cycles_are_collected():
    import gc
    import weakref

    class Problem:
        def __init__(self):
            self.response = solver.StaticEquilibriumResponse(
                2,
                dof.DofReduction(2, []),
                np.zeros(2),
                solver.NewtonOptions(gradient_tolerance=1e-12),
                self.make_energy,
                lambda u: FormulaForce(2, u, self.make_energy, self.partial),
            )

        def make_energy(self, p):
            return energy.QuadraticEnergy(sparse.eye(2, format="csc"), -p)

        def partial(self, p, u):
            return -sparse.eye(2, format="csc")

    problem = Problem()
    owner = weakref.ref(problem)
    response = problem.response
    del problem
    gc.collect()
    assert owner() is not None  # A live response must keep its callback owner alive.
    u, J = response.eval_value_and_jacobian(np.array([2.0, 3.0]))
    np.testing.assert_allclose(u, [2.0, 3.0])
    np.testing.assert_allclose(J.toarray(), np.eye(2))
    del response
    gc.collect()
    assert owner() is None


def test_gc_handles_uninitialized_and_failed_construction():
    import gc
    import pytest

    instance = solver.StaticEquilibriumResponse.__new__(
        solver.StaticEquilibriumResponse
    )
    gc.collect()
    del instance
    with pytest.raises(ValueError):
        solver.StaticEquilibriumResponse(
            2,
            dof.DofReduction(2, []),
            np.zeros(3),
            solver.NewtonOptions(gradient_tolerance=1e-12),
            lambda p: None,
            lambda u: FormulaForce(2, u, lambda p: None, lambda p, u: None),
        )
    gc.collect()


def test_fast_products_rectangular_fixed_dofs_and_recovery():
    import pytest

    K = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, -0.4], [0.5, -0.4, 2.0]])
    A = np.array([[1.0, -0.5], [0.3, 2.0], [-0.7, 1.2]])
    calls = []
    mode = "ok"

    def partial(p, u):
        calls.append(u.copy())
        if mode == "throw":
            raise RuntimeError("partial failed")
        if mode == "nan":
            return sparse.csc_matrix(np.full((3, 2), np.nan))
        return sparse.csc_matrix(-A)

    response = solver.StaticEquilibriumResponse(
        2,
        dof.DofReduction(3, [1]),
        np.array([0.0, 0.8, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        lambda p: energy.QuadraticEnergy(sparse.csc_matrix(K), -A @ p),
        lambda u: FormulaForce(
            2,
            u,
            lambda p: energy.QuadraticEnergy(sparse.csc_matrix(K), -A @ p),
            partial,
        ),
    )
    p = np.array([0.6, -1.1])
    v, w = np.array([0.3, -0.7]), np.array([0.4, 10.0, -0.8])
    J = response.eval_jacobian(p)
    calls.clear()
    jv = response.eval_jvp(p, v)
    vjp = response.eval_vjp(p, w)
    assert len(calls) == 2
    np.testing.assert_allclose(jv, J @ v, atol=1e-12)
    np.testing.assert_allclose(vjp, J.T @ w, atol=1e-12)
    np.testing.assert_allclose(w @ jv, v @ vjp, atol=1e-12)
    assert jv[1] == 0.0
    np.testing.assert_array_equal(
        response.eval_vjp(p, np.array([0.0, 1.0, 0.0])), np.zeros(2)
    )
    h = 1e-5
    numerical = np.array(
        [
            w
            @ (response.eval_value(p + h * e) - response.eval_value(p - h * e))
            / (2 * h)
            for e in np.eye(2)
        ]
    )
    np.testing.assert_allclose(vjp, numerical, atol=1e-9)
    for mode, error in [("throw", RuntimeError), ("nan", RuntimeError)]:
        with pytest.raises(error):
            response.eval_jvp(p, v)
        with pytest.raises(error):
            response.eval_vjp(p, w)
    mode = "ok"
    np.testing.assert_allclose(response.eval_vjp(p, w), vjp, atol=1e-12)
    with pytest.raises(ValueError):
        response.eval_jvp(p, np.ones(3))
    with pytest.raises(ValueError):
        response.eval_vjp(p, np.full(3, np.nan))


def test_fully_fixed_products_do_not_request_partials():
    def forbidden(*args):
        raise AssertionError("fully fixed state needs no derivative solve")

    response = solver.StaticEquilibriumResponse(
        2,
        dof.DofReduction(3, [0, 1, 2]),
        np.array([0.2, -0.4, 0.7]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        lambda p: energy.QuadraticEnergy(sparse.eye(3, format="csc")),
        lambda u: FormulaForce(
            2,
            u,
            lambda p: energy.QuadraticEnergy(sparse.eye(3, format="csc")),
            forbidden,
        ),
    )
    p = np.ones(2)
    np.testing.assert_array_equal(response.eval_jvp(p, p), np.zeros(3))
    np.testing.assert_array_equal(response.eval_vjp(p, np.ones(3)), np.zeros(2))


def test_products_use_exact_indefinite_stiffness():
    K = sparse.diags([2.0, -4.0], format="csc")
    response = solver.StaticEquilibriumResponse(
        2,
        dof.DofReduction(2, []),
        np.zeros(2),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        lambda p: energy.QuadraticEnergy(K, -p),
        lambda u: FormulaForce(
            2,
            u,
            lambda p: energy.QuadraticEnergy(K, -p),
            lambda p, u: -sparse.eye(2, format="csc"),
        ),
    )
    # At this stationary point Newton need not move; derivatives must retain the negative curvature.
    p, direction = np.zeros(2), np.array([3.0, 2.0])
    np.testing.assert_allclose(response.eval_jvp(p, direction), [1.5, -0.5])
    np.testing.assert_allclose(response.eval_vjp(p, direction), [1.5, -0.5])


def test_response_factory_direct_products_and_invalid_results():
    import pytest

    K = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, -0.4], [0.5, -0.4, 2.0]])
    A = np.array([[1.0, -0.5], [0.3, 2.0], [-0.7, 1.2]])
    states = []
    mode = "ok"

    class Force(energy.FixedPatternResponse):
        def __init__(self, u):
            super().__init__(np.ones((3, 2)))
            self.u = u.copy()

        def eval_impl(self, p, need_jacobian):
            if need_jacobian:
                raise AssertionError("explicit B requested")
            return K @ self.u - A @ p

        def eval_jvp_impl(self, p, v):
            return -A @ v

        def eval_vjp_impl(self, p, w):
            return -A.T @ w

    def make_response(u):
        states.append(u.copy())
        if mode == "null":
            return None
        if mode == "dimensions":
            return FormulaForce(1, u, make_energy, lambda p, u: np.zeros((3, 1)))
        if mode == "throw":
            raise RuntimeError("factory failed")
        return Force(u)

    def make_energy(p):
        return energy.QuadraticEnergy(sparse.csc_matrix(K), -A @ p)

    eq = solver.StaticEquilibriumResponse(
        2,
        dof.DofReduction(3, [1]),
        np.array([0.0, 0.8, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        make_energy,
        make_response,
    )
    p, v, w = np.array([0.6, -1.1]), np.array([0.3, -0.7]), np.array([0.4, 7.0, -0.8])
    u = eq.eval_value(p)
    assert not states
    np.testing.assert_allclose(make_energy(p).eval_gradient(u), Force(u).eval_value(p))
    J = np.zeros((3, 2))
    J[[0, 2]] = np.linalg.solve(K[np.ix_([0, 2], [0, 2])], A[[0, 2]])
    np.testing.assert_allclose(eq.eval_jvp(p, v), J @ v, atol=1e-12)
    np.testing.assert_allclose(eq.eval_vjp(p, w), J.T @ w, atol=1e-12)
    np.testing.assert_allclose(states[-1], u)
    h = 1e-5
    np.testing.assert_allclose(
        eq.eval_jvp(p, v),
        (eq.eval_value(p + h * v) - eq.eval_value(p - h * v)) / (2 * h),
        atol=1e-9,
    )
    with pytest.raises(AssertionError, match="explicit B"):
        eq.eval_jacobian(p)
    for mode, error in [
        ("null", ValueError),
        ("dimensions", ValueError),
        ("throw", RuntimeError),
    ]:
        for call in [
            lambda: eq.eval_jvp(p, v),
            lambda: eq.eval_vjp(p, w),
            lambda: eq.eval_jacobian(p),
        ]:
            with pytest.raises(error):
                call()
    mode = "ok"
    np.testing.assert_allclose(eq.eval_vjp(p, w), J.T @ w, atol=1e-12)
    # The Torch bridge must reach these products rather than requesting B.
    torch = pytest.importorskip("torch")
    from litepgo import torch as bridge

    x = torch.tensor(p, dtype=torch.float64, requires_grad=True)
    (bridge.response(eq, x) * torch.tensor(w)).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), J.T @ w, atol=1e-12)
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(x.detach(), torch.tensor(v))
        tangent = torch.autograd.forward_ad.unpack_dual(
            bridge.response(eq, dual)
        ).tangent
        np.testing.assert_allclose(tangent.numpy(), J @ v, atol=1e-12)


def test_factory_response_dynamic_pattern_is_refreshed():
    class Force(energy.DynamicPatternResponse):
        def __init__(self, u):
            super().__init__(2, 2)
            self.u = u.copy()

        def jacobian_pattern_at(self, p):
            return np.array([[1.0, 0.0], [float(p[0] > 0), 1.0]])

        def eval_impl(self, p, need_jacobian):
            value = self.u - np.array([p[0], p[1] + max(p[0], 0.0) ** 2])
            J = np.array([[-1.0, 0.0], [-2 * max(p[0], 0.0), -1.0]])
            return (value, J) if need_jacobian else value

    eq = solver.StaticEquilibriumResponse(
        2,
        dof.DofReduction(2, []),
        np.zeros(2),
        solver.NewtonOptions(gradient_tolerance=1e-12),
        lambda p: energy.QuadraticEnergy(
            sparse.eye(2, format="csc"), -np.array([p[0], p[1] + max(p[0], 0.0) ** 2])
        ),
        Force,
    )
    for p in [np.array([0.3, 0.4]), np.array([-0.3, 0.4])]:
        J = np.array([[1.0, 0.0], [2 * max(p[0], 0.0), 1.0]])
        np.testing.assert_allclose(eq.eval_jacobian(p).toarray(), J)
        np.testing.assert_allclose(eq.eval_jvp(p, p), J @ p)
        np.testing.assert_allclose(eq.eval_vjp(p, p), J.T @ p)
