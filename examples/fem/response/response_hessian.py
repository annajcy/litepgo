"""Triangle P1 sampling, a nonlinear Response, and the exact energy chain rule.

Run: PYTHONPATH=build/release/python uv run --no-sync python examples/fem/response/response_hessian.py
"""

import numpy as np
from litepgo import energy, fem


class TriangleSample(energy.FixedPatternResponse):
    def __init__(self, xi):
        self.N = fem.TriangleP1Basis.eval_N(xi)
        # An empty 3x3 pattern explicitly enables the zero Hessian of a linear map.
        super().__init__(np.ones((1, 3)), hessian_pattern=np.zeros((3, 3)))

    def eval_impl(self, c, need_jacobian):
        value = np.array([self.N @ c])
        return (value, self.N[None, :]) if need_jacobian else value

    def eval_hessian_impl(self, c, weights):
        return np.zeros((3, 3))


def main():
    xi = np.array([0.2, 0.3])
    c = np.array([-0.2, 0.1, 0.4])
    sample = TriangleSample(xi)
    response = energy.ComposedResponse(energy.ExpResponse(1), sample)
    response.update_patterns(c)
    theta = response.eval_value(c)[0]
    J = response.eval_jacobian(c).toarray()
    H = response.eval_hessian(c, weights=np.ones(1)).toarray()
    NN = np.outer(sample.N, sample.N)
    np.testing.assert_allclose(H, theta * NN)

    # W(theta) = theta^2 / 2: W' = theta, W'' = 1.
    energy_H = J.T @ J + response.eval_hessian(c, weights=np.array([theta])).toarray()
    np.testing.assert_allclose(energy_H, 2 * theta**2 * NN)
    print(f"theta={theta:.6f}, response H00={H[0, 0]:.6f}, energy H00={energy_H[0, 0]:.6f}")


if __name__ == "__main__":
    main()
