"""Evaluate Tet P1 and reconstruct u(r,s,t) = 2*r + 3*s - t with a field.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/basis/tet_p1_basis.py
"""

import numpy as np
from litepgo import fem


def target(xi):
    r, s, t = xi
    return 2 * r + 3 * s - t


def main():
    B = fem.TetP1Basis
    xi = np.array([0.25, 0.25, 0.25])
    assert B.num_basis_functions == 4
    N = B.eval_N(xi)
    dN = B.eval_dN_dxi(xi)
    d2N = B.eval_d2N_dxi2(xi)

    np.testing.assert_allclose(N, np.full(4, 0.25))
    np.testing.assert_allclose(dN, [[-1, -1, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(d2N, np.zeros((4, 6)))
    print(f"TetP1: functions={B.num_basis_functions}, verified=1")

    # Tet P1 coefficients are target values sampled at its four vertices.
    coefficients = np.array([target(B.coefficient_info(i).point) for i in range(B.num_basis_functions)])[:, None]
    field = fem.InterpolatedResponse[B](coefficients)
    points = np.array([[0.25, 0.25, 0.25], [0.2, 0.3, 0.1], [0.1, 0.1, 0.2]])
    for point in points:
        np.testing.assert_allclose(field.eval_value(point)[0], target(point))
    print(f"TetP1 field: samples={len(points)}, verified=1")


if __name__ == "__main__":
    main()
