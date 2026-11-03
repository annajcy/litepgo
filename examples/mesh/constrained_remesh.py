"""Remesh a square with a fixed interior vertex and protected boundary edges."""

from collections import Counter

import numpy as np

from litepgo import mesh


def boundary_segments(data):
    positions = data.positions
    incidence = Counter(
        tuple(sorted((triangle[corner], triangle[(corner + 1) % 3])))
        for triangle in data.cells
        for corner in range(3)
    )
    return {
        tuple(sorted((tuple(positions[a]), tuple(positions[b]))))
        for (a, b), count in incidence.items()
        if count == 1
    }


def main():
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.2, 0.2, 0]])
    cells = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
    source = mesh.TriangleMesh(positions, cells)
    result = mesh.isotropic_remesh(
        source, target_edge_length=0.8, num_iter=5, constrained_vertices=[4]
    )

    assert np.count_nonzero(np.all(result.positions == positions[4], axis=1)) == 1
    assert boundary_segments(result) == boundary_segments(source)
    assert not mesh.has_self_intersections(result)
    print(
        f"{len(cells)} -> {result.topology.num_cells} triangles. "
        "Fixed point, boundary, and self-intersection checks passed."
    )


if __name__ == "__main__":
    main()
