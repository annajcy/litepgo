"""Build Hermite coefficients from sample descriptions for u(xi) = sum(xi**2).

Run: uv run --no-sync python examples/fem/basis/hermite_interpolation.py
"""

import numpy as np
from litepgo import fem


def target(point):
    return point @ point


def sample(info):
    point = np.atleast_1d(info.point)
    degree = sum(info.derivative_order)
    if degree == 0:
        return target(point)
    if degree == 1:
        return 2 * point[info.derivative_order.index(1)]
    return 0.0


def main():
    for B, xi, name in [
        (fem.CubicHermiteBasis, 0.5, "CubicHermite"),
        (fem.QuadHermiteBasis, np.full(2, 0.5), "QuadHermite"),
        (fem.HexHermiteBasis, np.full(3, 0.5), "HexHermite"),
    ]:
        coefficients = np.array([sample(B.coefficient_info(i)) for i in range(B.num_basis_functions)])
        value = coefficients @ B.eval_N(xi)
        point = np.atleast_1d(xi)
        gradient = coefficients @ B.eval_dN_dxi(xi)
        hessian = coefficients @ B.eval_d2N_dxi2(xi)
        np.testing.assert_allclose(value, target(point))
        np.testing.assert_allclose(gradient, 2 * point)
        np.testing.assert_allclose(
            hessian, [2 if i == j else 0 for i in range(len(point)) for j in range(i, len(point))]
        )
        print(f"{name}: value={value:.2f}, verified=1")

        if len(point) > 1:
            Field = fem.InterpolatedResponse
            field = Field[B](coefficients[:, None])
            points = [0.2 + 0.15 * np.arange(len(point)) + 0.1 * i for i in range(3)]
            for query in points:
                np.testing.assert_allclose(field.eval_value(query)[0], target(query))
            print(f"{name} field: samples={len(points)}, verified=1")


if __name__ == "__main__":
    main()
