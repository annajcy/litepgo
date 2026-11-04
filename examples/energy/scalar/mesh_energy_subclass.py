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
        H = (
            (k[:, None, None] * np.array([[1.0, -1.0], [-1.0, 1.0]])).reshape(-1, 2)
            if need_hessian
            else None
        )
        return 0.5 * k * stretch**2, g, H


def main():
    mapping = dof.StencilDofMap(4, [[0, 1], [1, 2], [2, 3]], 1)
    stiffness = np.array([2.0, 3.0, 4.0])
    lengths = np.array([1.0, 2.0, 1.0])

    potential = SpringEnergy(mapping, stiffness, lengths)
    initial = np.array([0.0, 0.0, 0.0, 7.0])
    result = solver.newton_solve(
        potential,
        dof.DofReduction(4, [0, 3]),
        initial,
        solver.NewtonOptions(gradient_tolerance=1e-10),
    )
    if result.status != solver.NewtonStatus.Converged:
        raise RuntimeError("Fixed-end equilibrium failed")
    print(
        "mesh energy subclass: 3 springs, shared-node assembly and fixed-end equilibrium"
    )


if __name__ == "__main__":
    main()
