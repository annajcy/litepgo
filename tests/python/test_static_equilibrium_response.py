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
    response_evaluation = response.evaluate(p)
    J = response_evaluation.jacobian()
    u = response_evaluation.value
    np.testing.assert_allclose(u, [0.4, 3.0])
    np.testing.assert_allclose(J.toarray(), [[0, 0], [0, 1]])
    for c in range(2):
        delta = np.eye(2)[c] * 1e-4
        numerical = (
            response.evaluate(p + delta).value - response.evaluate(p - delta).value
        ) / 2e-4
        np.testing.assert_allclose(J.toarray()[:, c], numerical, atol=1e-10)
    assert len(calls) == 1


def test_bound_callbacks_are_kept_alive_without_an_owner_cycle():
    import gc
    import weakref

    class Problem:
        def make_equilibrium(self):
            response = solver.StaticEquilibriumResponse(
                2,
                dof.DofReduction(2, []),
                np.zeros(2),
                solver.NewtonOptions(gradient_tolerance=1e-12),
                self.make_energy,
                lambda u: FormulaForce(2, u, self.make_energy, self.partial),
            )
            self.response = weakref.ref(response)
            return response

        def make_energy(self, p):
            return energy.QuadraticEnergy(sparse.eye(2, format="csc"), -p)

        def partial(self, p, u):
            return -sparse.eye(2, format="csc")

    problem = Problem()
    owner = weakref.ref(problem)
    response = problem.make_equilibrium()
    assert problem.response() is response
    del problem
    gc.collect()
    assert owner() is not None  # A live response must keep its callback owner alive.
    response_evaluation = response.evaluate(np.array([2.0, 3.0]))
    J = response_evaluation.jacobian()
    u = response_evaluation.value
    np.testing.assert_allclose(u, [2.0, 3.0])
    np.testing.assert_allclose(J.toarray(), np.eye(2))
    del response
    gc.collect()
    assert owner() is not None  # The retained evaluation still owns the callbacks.
    del response_evaluation
    gc.collect()
    assert owner() is None


def test_global_callback_cycle_is_collected():
    import gc
    import weakref

    def make_cycle():
        stiffness = sparse.eye(1, format="csc")
        scope = {"energy": energy, "stiffness": stiffness}
        exec(
            "def make_energy(p):\n"
            "    return energy.QuadraticEnergy(stiffness, -p)\n"
            "def make_response(u):\n"
            "    return energy.AffineResponse(-stiffness, u)\n",
            scope,
        )
        equilibrium = solver.StaticEquilibriumResponse(
            1,
            dof.DofReduction(1, []),
            np.zeros(1),
            solver.NewtonOptions(gradient_tolerance=1e-12),
            scope["make_energy"],
            scope["make_response"],
        )
        scope["equilibrium"] = equilibrium
        gc.collect()
        equilibrium_evaluation = equilibrium.evaluate(np.array([2.0]))
        jacobian = equilibrium_evaluation.jacobian()
        u = equilibrium_evaluation.value
        np.testing.assert_allclose(u, [2.0])
        np.testing.assert_allclose(jacobian.toarray(), [[1.0]])
        return [weakref.ref(scope[name]) for name in ("equilibrium", "make_energy", "make_response")]

    refs = make_cycle()
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_invalid_initial_state_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        solver.StaticEquilibriumResponse(
            2,
            dof.DofReduction(2, []),
            np.zeros(3),
            solver.NewtonOptions(gradient_tolerance=1e-12),
            lambda p: None,
            lambda u: FormulaForce(2, u, lambda p: None, lambda p, u: None),
        )


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
    J = response.evaluate(p).jacobian()
    calls.clear()
    jv = response.evaluate(p).jvp(v)
    vjp = response.evaluate(p).vjp(w)
    assert len(calls) == 2
    np.testing.assert_allclose(jv, J @ v, atol=1e-12)
    np.testing.assert_allclose(vjp, J.T @ w, atol=1e-12)
    np.testing.assert_allclose(w @ jv, v @ vjp, atol=1e-12)
    assert jv[1] == 0.0
    np.testing.assert_array_equal(response.evaluate(p).vjp(np.array([0.0, 1.0, 0.0])), np.zeros(2))
    h = 1e-5
    numerical = np.array([w @ (response.evaluate(p + h * e).value - response.evaluate(p - h * e).value) / (2 * h) for e in np.eye(2)])
    np.testing.assert_allclose(vjp, numerical, atol=1e-9)
    for mode, error in [("throw", RuntimeError), ("nan", RuntimeError)]:
        with pytest.raises(error):
            response.evaluate(p).jvp(v)
        with pytest.raises(error):
            response.evaluate(p).vjp(w)
    mode = "ok"
    np.testing.assert_allclose(response.evaluate(p).vjp(w), vjp, atol=1e-12)
    with pytest.raises(ValueError):
        response.evaluate(p).jvp(np.ones(3))
    with pytest.raises(ValueError):
        response.evaluate(p).vjp(np.full(3, np.nan))


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
    np.testing.assert_array_equal(response.evaluate(p).jvp(p), np.zeros(3))
    np.testing.assert_array_equal(response.evaluate(p).vjp(np.ones(3)), np.zeros(2))


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
    np.testing.assert_allclose(response.evaluate(p).jvp(direction), [1.5, -0.5])
    np.testing.assert_allclose(response.evaluate(p).vjp(direction), [1.5, -0.5])


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

        def jvp_at(self, p, v):
            return -A @ v

        def vjp_at(self, p, w):
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
    u = eq.evaluate(p).value
    assert not states
    np.testing.assert_allclose(make_energy(p).eval_gradient(u), Force(u).evaluate(p).value)
    J = np.zeros((3, 2))
    J[[0, 2]] = np.linalg.solve(K[np.ix_([0, 2], [0, 2])], A[[0, 2]])
    np.testing.assert_allclose(eq.evaluate(p).jvp(v), J @ v, atol=1e-12)
    np.testing.assert_allclose(eq.evaluate(p).vjp(w), J.T @ w, atol=1e-12)
    np.testing.assert_allclose(states[-1], u)
    h = 1e-5
    np.testing.assert_allclose(eq.evaluate(p).jvp(v), (eq.evaluate(p + h * v).value - eq.evaluate(p - h * v).value) / (2 * h), atol=1e-09)
    with pytest.raises(AssertionError, match="explicit B"):
        eq.evaluate(p).jacobian()
    for mode, error in [
        ("null", ValueError),
        ("dimensions", ValueError),
        ("throw", RuntimeError),
    ]:
        for call in [
            lambda: eq.evaluate(p).jvp(v),
            lambda: eq.evaluate(p).vjp(w),
            lambda: eq.evaluate(p).jacobian(),
        ]:
            with pytest.raises(error):
                call()
    mode = "ok"
    np.testing.assert_allclose(eq.evaluate(p).vjp(w), J.T @ w, atol=1e-12)
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
        np.testing.assert_allclose(eq.evaluate(p).jacobian().toarray(), J)
        np.testing.assert_allclose(eq.evaluate(p).jvp(p), J @ p)
        np.testing.assert_allclose(eq.evaluate(p).vjp(p), J.T @ p)
