"""Weighted energies in reduced coordinates x = T z + offset.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/energy/scalar/energy_composition.py
"""

import numpy as np
import scipy.sparse as sp

from litepgo.energy import (
    AffineMappedEnergy,
    HessianKind,
    QuadraticEnergy,
    VertexPullingEnergy,
    WeightedSumEnergy,
)


def main():
    target = np.array([1.0, 2.0, 3.0])
    regularizer = QuadraticEnergy(sp.eye(3, format="csc"))
    pulling = VertexPullingEnergy(
        3, np.array([[0, 1, 2]], dtype=np.int32), target[None, :], 1.0
    )
    # All terms share the same DOFs; weights scale their values and derivatives.
    total = WeightedSumEnergy([(regularizer, 0.5), (pulling, 2.0)])
    T = sp.csc_matrix([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    offset = np.array([0.0, 0.0, 1.0])
    mapped = AffineMappedEnergy(total, T, offset)
    z = np.array([0.2, -0.3])
    # Joint evaluation prepares structure and returns independent results.
    value, gradient, hessian = mapped.eval(z, gradient=True, hessian=HessianKind.Exact)
    x = T @ z + offset
    np.testing.assert_allclose(value, 0.25 * (x @ x) + np.sum((x - target) ** 2))
    np.testing.assert_allclose(gradient, T.T @ (2.5 * x - 2.0 * target))
    np.testing.assert_allclose(hessian.toarray(), (2.5 * T.T @ T).toarray())
    print(f"energy={value:g}, gradient={gradient}, Hessian=\n{hessian.toarray()}")


if __name__ == "__main__":
    main()
