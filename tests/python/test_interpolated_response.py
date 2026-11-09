import numpy as np
import pytest
from litepgo import energy, fem

BASES = [
    (fem.LinearLagrangeBasis, 1), (fem.CubicHermiteBasis, 1),
    (fem.TriangleP1Basis, 2), (fem.QuadHermiteBasis, 2),
    (fem.TetP1Basis, 3), (fem.HexQ1Basis, 3), (fem.HexHermiteBasis, 3),
]


@pytest.mark.parametrize("basis,d", BASES)
def test_spatial_derivatives_and_snapshot(basis, d):
    C = np.random.default_rng(42).normal(size=(basis.num_basis_functions, 2))
    saved = C.copy()
    r = fem.InterpolatedResponse[basis](coefficients=C)
    assert isinstance(r, energy.Response)
    assert (r.num_inputs, r.num_outputs, r.has_hessian) == (d, 2, True)
    C[:] = 0
    x = np.linspace(.2, .4, d)
    w = np.array([1.3, -.7])
    v = np.linspace(-.4, .6, d)
    r_evaluation = r.evaluate(x)
    J = r_evaluation.jacobian()
    y = r_evaluation.value
    np.testing.assert_allclose(y, saved.T @ basis.eval_N(x[0] if d == 1 else x))
    eps = 1e-5
    numerical = np.column_stack([(r.evaluate(x + eps * e).value - r.evaluate(x - eps * e).value) / (2 * eps) for e in np.eye(d)])
    np.testing.assert_allclose(J.toarray(), numerical, atol=1e-8)
    H = r.evaluate(x).hessian(w)
    numerical_H = np.column_stack([(r.evaluate(x + eps * e).vjp(w) - r.evaluate(x - eps * e).vjp(w)) / (2 * eps) for e in np.eye(d)])
    np.testing.assert_allclose(H.toarray(), numerical_H, atol=1e-8)
    np.testing.assert_allclose(H.toarray(), H.toarray().T)
    np.testing.assert_allclose(r.evaluate(x).jvp(v), J @ v)
    np.testing.assert_allclose(r.evaluate(x).vjp(w), J.T @ w)
    np.testing.assert_array_equal(r.evaluate(x).hessian(np.zeros(2)).toarray(), np.zeros((d, d)))
    original = H.toarray().copy()
    r.evaluate(x + 0.3).hessian(-w)
    np.testing.assert_array_equal(H.toarray(), original)
    outside = np.full(d, 1.4)
    np.testing.assert_allclose(r.evaluate(outside).value, saved.T @ basis.eval_N(outside[0] if d == 1 else outside))


def test_tet_geometry_and_invalid_inputs():
    cls = fem.InterpolatedResponse[fem.TetP1Basis]
    X = np.vstack([np.zeros(3), np.diag([2., 3., 4.])])
    r = cls(X)
    np.testing.assert_allclose(r.evaluate(np.array([0.2, 0.3, 0.1])).value, [0.4, 0.9, 0.4])
    np.testing.assert_array_equal(r.evaluate(np.zeros(3)).jacobian().toarray(), np.diag([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(r.evaluate(np.zeros(3)).hessian(np.ones(3)).toarray(), np.zeros((3, 3)))
    for invalid in [np.zeros((3, 2)), np.empty((4, 0)), np.full((4, 2), np.nan), np.full((4, 2), np.inf)]:
        with pytest.raises(ValueError):
            cls(invalid)
    for x in [np.ones(2), np.full(3, np.nan)]:
        with pytest.raises(ValueError):
            r.evaluate(x).value
    with pytest.raises(ValueError):
        r.evaluate(np.zeros(3)).hessian(np.ones(2))
    with pytest.raises(ValueError):
        r.evaluate(np.zeros(3)).jvp(np.ones(2))
