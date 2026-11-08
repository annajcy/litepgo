"""Integrate polynomials and constants on reference elements.

Run: uv run --no-sync python examples/fem/quadrature/quadrature_integration.py
"""

import numpy as np
from litepgo import fem


def run(Q, name, f, expected):
    points = Q.points()
    weights = Q.weights()
    integral = weights @ f(points)
    np.testing.assert_allclose(integral, expected, rtol=0, atol=1e-12)
    print(f"{name}: points={Q.num_points}, integral={integral:.6f}, verified=1")


def main():
    def squared_norm(xi):
        return np.sum(xi**2, axis=1)

    run(fem.GaussLegendre1D[2], "GaussLegendre1D[2]", squared_norm, 1 / 3)
    run(fem.GaussLegendre2D[2], "GaussLegendre2D[2]", squared_norm, 2 / 3)
    run(fem.GaussLegendre3D[2], "GaussLegendre3D[2]", squared_norm, 1)

    def constant(xi):
        return np.ones(xi.shape[0])

    run(fem.TriangleCentroidQuadrature, "TriangleCentroid", constant, 1 / 2)
    run(fem.TetCentroidQuadrature, "TetCentroid", constant, 1 / 6)


if __name__ == "__main__":
    main()
