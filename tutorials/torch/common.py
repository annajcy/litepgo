"""Shared geometry only; constitutive laws and differentiation stay in the lessons."""

import numpy as np
from litepgo import dof, energy, fem, mesh


def make_geometry():
    X = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
    topology = mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))
    mapping = dof.VertexDofMap(topology, 3)
    geometry = [fem.InterpolatedResponse[fem.TetP1Basis](mapping.gather(e, X.ravel()))
                for e in range(topology.num_cells)]
    plastic = [energy.AffineResponse(np.zeros((9, 3)), np.eye(3).ravel())] * topology.num_cells
    return X, topology, mapping, geometry, plastic
