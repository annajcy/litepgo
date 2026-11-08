"""Fixed coefficients, spatial queries and exact weighted spatial Hessians."""
import numpy as np
from litepgo import fem


def main():
    X = np.array([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
    geometry = fem.InterpolatedResponse[fem.TetP1Basis](X)
    xi = np.array([.2, .3, .1])
    value, J = geometry.eval_value_and_jacobian(xi)
    np.testing.assert_allclose(value, [.4, .9, .4])
    np.testing.assert_allclose(J.toarray(), np.diag([2., 3., 4.]))
    np.testing.assert_array_equal(geometry.eval_hessian(xi, np.ones(3)).toarray(), np.zeros((3, 3)))

    # Two components: f(xi)=xi_1*xi_2 and g(xi)=xi_2*xi_3.
    B = fem.HexQ1Basis
    C = np.array([[p[0]*p[1], p[1]*p[2]]
                  for p in (B.coefficient_info(i).point for i in range(B.num_basis_functions))])
    response = fem.InterpolatedResponse[B](C)
    weights = np.array([2., -3.])
    H = response.eval_hessian(xi, weights)
    np.testing.assert_allclose(H.toarray(), [[0., 2., 0.], [2., 0., -3.], [0., -3., 0.]])
    print("spatial geometry and weighted Hessian: verified")


if __name__ == "__main__":
    main()
