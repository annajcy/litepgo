from parametric_fixtures import FormulaModel
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
        FormulaModel(2, (dof.DofReduction(2, [0])).num_dofs, make_energy, partial),
        dof.DofReduction(2, [0]),
        np.array([0.4, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
    )
    p = np.array([2.0, 3.0])
    u, J = response.eval(p, need_jacobian=True)
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
                FormulaModel(
                    2,
                    (dof.DofReduction(2, [])).num_dofs,
                    self.make_energy,
                    self.partial,
                ),
                dof.DofReduction(2, []),
                np.zeros(2),
                solver.NewtonOptions(gradient_tolerance=1e-12),
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
    u, J = response.eval(np.array([2.0, 3.0]), need_jacobian=True)
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
            FormulaModel(
                2, (dof.DofReduction(2, [])).num_dofs, lambda p: None, lambda p, u: None
            ),
            dof.DofReduction(2, []),
            np.zeros(3),
            solver.NewtonOptions(gradient_tolerance=1e-12),
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
        FormulaModel(
            2,
            (dof.DofReduction(3, [1])).num_dofs,
            lambda p: energy.QuadraticEnergy(sparse.csc_matrix(K), -A @ p),
            partial,
        ),
        dof.DofReduction(3, [1]),
        np.array([0.0, 0.8, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
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
        FormulaModel(
            2,
            (dof.DofReduction(3, [0, 1, 2])).num_dofs,
            lambda p: energy.QuadraticEnergy(sparse.eye(3, format="csc")),
            forbidden,
        ),
        dof.DofReduction(3, [0, 1, 2]),
        np.array([0.2, -0.4, 0.7]),
        solver.NewtonOptions(gradient_tolerance=1e-12),
    )
    p = np.ones(2)
    np.testing.assert_array_equal(response.eval_jvp(p, p), np.zeros(3))
    np.testing.assert_array_equal(response.eval_vjp(p, np.ones(3)), np.zeros(2))


def test_products_use_exact_indefinite_stiffness():
    K = sparse.diags([2.0, -4.0], format="csc")
    response = solver.StaticEquilibriumResponse(
        FormulaModel(
            2,
            (dof.DofReduction(2, [])).num_dofs,
            lambda p: energy.QuadraticEnergy(K, -p),
            lambda p, u: -sparse.eye(2, format="csc"),
        ),
        dof.DofReduction(2, []),
        np.zeros(2),
        solver.NewtonOptions(gradient_tolerance=1e-12),
    )
    # At this stationary point Newton need not move; derivatives must retain the negative curvature.
    p, direction = np.zeros(2), np.array([3.0, 2.0])
    np.testing.assert_allclose(response.eval_jvp(p, direction), [1.5, -0.5])
    np.testing.assert_allclose(response.eval_vjp(p, direction), [1.5, -0.5])


def test_removed_callback_constructor_is_rejected():
    import pytest

    with pytest.raises(TypeError):
        solver.StaticEquilibriumResponse(
            2,
            dof.DofReduction(2, []),
            np.zeros(2),
            solver.NewtonOptions(gradient_tolerance=1e-12),
            lambda p: None,
            lambda p, u: None,
        )
