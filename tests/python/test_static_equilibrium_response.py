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

    response = energy.StaticEquilibriumResponse(
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
