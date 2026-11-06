"""Fixed reference points, variable coefficients, and reference-derivative sampling."""
import numpy as np
from litepgo import energy, fem


def main():
    B = fem.HexQ1Basis
    points = np.array([[.2, .3, .4], [.6, .7, .8]])
    # Two components: f(xi)=xi_1*xi_2, g(xi)=xi_2*xi_3.
    C = np.array([[p[0]*p[1], p[1]*p[2]]
                  for p in (B.coefficient_info(a).point for a in range(B.num_basis_functions))])
    c = C.ravel()  # basis coefficient first, then component
    values = fem.value_sampling_response[B](points, components=2)
    gradients = fem.gradient_sampling_response[B](points, components=2)
    hessians = fem.hessian_sampling_response[B](points, components=2)
    assert all(type(r) is energy.AffineResponse for r in (values, gradients, hessians))
    np.testing.assert_allclose(values.eval_value(c).reshape(2, 2), [[.06, .12], [.42, .56]])
    # Per point: xi direction first, then component.
    np.testing.assert_allclose(gradients.eval_value(c).reshape(2, 3, 2),
                               [[[.3, 0], [.2, .4], [0, .3]],
                                [[.7, 0], [.6, .8], [0, .7]]])
    # Per point: (11,12,13,22,23,33), then component; no factor of two.
    expected = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [0, 1], [0, 0]])
    np.testing.assert_allclose(hessians.eval_value(c).reshape(2, 6, 2), np.tile(expected, (2, 1, 1)))
    A = gradients.eval_jacobian(c)
    direction = np.ones_like(c)
    weights = np.arange(gradients.num_outputs, dtype=float)
    np.testing.assert_allclose(gradients.eval_jvp(c, direction), A @ direction)
    np.testing.assert_allclose(gradients.eval_vjp(c, weights), A.T @ weights)
    # The sampled spatial Hessian is nonzero; its derivative twice with respect to c is zero.
    assert hessians.eval_hessian(c, np.ones(hessians.num_outputs)).nnz == 0
    print("coefficient value, gradient and Hessian sampling: verified")


if __name__ == "__main__":
    main()
