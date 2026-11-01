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
        2, dof.DofReduction(2, [0]), np.array([0.4, 0.0]),
        solver.NewtonOptions(gradient_tolerance=1e-12), make_energy, partial,
    )
    p = np.array([2.0, 3.0])
    u, J = response.eval(p, need_jacobian=True)
    np.testing.assert_allclose(u, [0.4, 3.0])
    np.testing.assert_allclose(J.toarray(), [[0, 0], [0, 1]])
    for c in range(2):
        delta = np.eye(2)[c] * 1e-4
        numerical = (response.eval_value(p + delta) - response.eval_value(p - delta)) / 2e-4
        np.testing.assert_allclose(J.toarray()[:, c], numerical, atol=1e-10)
    assert len(calls) == 1


def test_bound_callbacks_are_kept_alive_and_cycles_are_collected():
    import gc
    import weakref

    class Problem:
        def __init__(self):
            self.response = solver.StaticEquilibriumResponse(
                2, dof.DofReduction(2, []), np.zeros(2),
                solver.NewtonOptions(gradient_tolerance=1e-12), self.make_energy, self.partial,
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
    u, J = response.eval(np.array([2., 3.]), need_jacobian=True)
    np.testing.assert_allclose(u, [2., 3.])
    np.testing.assert_allclose(J.toarray(), np.eye(2))
    del response
    gc.collect()
    assert owner() is None


def test_gc_handles_uninitialized_and_failed_construction():
    import gc
    import pytest

    instance = solver.StaticEquilibriumResponse.__new__(solver.StaticEquilibriumResponse)
    gc.collect()
    del instance
    with pytest.raises(ValueError):
        solver.StaticEquilibriumResponse(
            2, dof.DofReduction(2, []), np.zeros(3),
            solver.NewtonOptions(gradient_tolerance=1e-12), lambda p: None, lambda p, u: None,
        )
    gc.collect()
