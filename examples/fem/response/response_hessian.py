"""Triangle P1 sampling, a nonlinear Response, and the exact energy chain rule.

Run: PYTHONPATH=build/release/python uv run --no-sync python examples/fem/response/response_hessian.py
"""

import numpy as np
from litepgo import energy, fem


def main():
    xi = np.array([0.2, 0.3])
    c = np.array([-0.2, 0.1, 0.4])
    N = fem.TriangleP1Basis.eval_N(xi)
    sample = energy.AffineResponse(N[None, :])
    response = energy.ComposedResponse(energy.ExpResponse(1), sample)
    theta = response.evaluate(c).value[0]
    J = response.evaluate(c).jacobian().toarray()
    H = response.evaluate(c).hessian(weights=np.ones(1)).toarray()
    NN = np.outer(N, N)
    np.testing.assert_allclose(H, theta * NN)

    # W(theta) = theta^2 / 2: W' = theta, W'' = 1.
    energy_H = J.T @ J + response.evaluate(c).hessian(weights=np.array([theta])).toarray()
    np.testing.assert_allclose(energy_H, 2 * theta**2 * NN)
    print(f"theta={theta:.6f}, response H00={H[0, 0]:.6f}, energy H00={energy_H[0, 0]:.6f}")


if __name__ == "__main__":
    main()
