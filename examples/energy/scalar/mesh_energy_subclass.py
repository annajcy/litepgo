"""Assemble a spring chain with a MeshEnergy subclass, then solve with fixed endpoints.

Run: uv run --no-sync python examples/energy/scalar/mesh_energy_subclass.py
"""

import numpy as np
from litepgo import dof, energy, solver


class SpringEnergy(energy.MeshEnergy):
    def __init__(self, mapping, stiffness, lengths):
        super().__init__(mapping, batch_size=2)
        self.stiffness, self.lengths = stiffness.copy(), lengths.copy()

    def eval_batch(self, ids, x, need_gradient, need_hessian):
        # Each row contains the two endpoints of one edge; ids select its data.
        k = self.stiffness[ids]
        stretch = x[:, 1] - x[:, 0] - self.lengths[ids]
        force = k * stretch
        g = np.stack([-force, force], axis=1) if need_gradient else None
        H = (k[:, None, None] * np.array([[1., -1.], [-1., 1.]])).reshape(-1, 2) if need_hessian else None
        return .5 * k * stretch**2, g, H


def main():
    mapping = dof.StencilDofMap(4, [[0, 1], [1, 2], [2, 3]], 1)
    stiffness = np.array([2., 3., 4.])
    lengths = np.array([1., 2., 1.])

    potential = SpringEnergy(mapping, stiffness, lengths)
    initial = np.array([0., 0., 0., 7.])
    value, g, H = potential.eval(initial, gradient=True, hessian=energy.HessianKind.Exact)
    # Independent global incidence-matrix check catches shared-node assembly errors.
    D = np.array([[-1., 1., 0., 0.], [0., -1., 1., 0.], [0., 0., -1., 1.]])
    stretch = D @ initial - lengths
    np.testing.assert_allclose(value, .5 * np.dot(stiffness, stretch**2))
    np.testing.assert_allclose(g, D.T @ (stiffness * stretch))
    np.testing.assert_allclose(H.toarray(), D.T @ np.diag(stiffness) @ D)

    result = solver.newton_solve(
        potential, dof.DofReduction(4, [0, 3]), initial,
        solver.NewtonOptions(gradient_tolerance=1e-10),
    )
    assert result.status == solver.NewtonStatus.Converged
    # Springs in series have equal tension; extension is distributed as force/k.
    tension = (initial[-1] - lengths.sum()) / (1 / stiffness).sum()
    expected = np.r_[0., np.cumsum(lengths + tension / stiffness)]
    np.testing.assert_allclose(result.x, expected, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(result.x[[0, 3]], initial[[0, 3]])
    assert np.max(np.abs(potential.eval_gradient(result.x)[1:3])) < 1e-10
    print("mesh energy subclass: 3 springs, shared-node assembly and fixed-end equilibrium verified")


if __name__ == "__main__":
    main()
