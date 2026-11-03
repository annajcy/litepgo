"""Shared tetrahedral input for the mesh optimization examples."""

from itertools import combinations

import numpy as np

from litepgo import mesh


def read_volume(path):
    volume = mesh.read_veg(path)
    if not isinstance(volume, mesh.TetMesh) or volume.topology.num_cells == 0:
        raise ValueError("expected a nonempty tetrahedral VEG mesh")
    edges = sorted({tuple(sorted(edge)) for cell in volume.cells for edge in combinations(cell, 2)})
    return volume, edges


def end_faces(rest):
    x = rest[:, 0]
    left = np.flatnonzero(np.isclose(x, x.min(), rtol=0, atol=1e-8 * np.ptp(x)))
    right = np.flatnonzero(np.isclose(x, x.max(), rtol=0, atol=1e-8 * np.ptp(x)))
    return left, right
