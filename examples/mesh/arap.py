"""Uniform-weight vertex ARAP, alternating local rotations and global positions.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/mesh/arap.py
"""

from pathlib import Path
import argparse
import numpy as np
from litepgo import dof, energy, mesh, solver


def rotations(x, rest, edges):
    covariance = np.zeros((len(x), 3, 3))
    for a, b in edges:
        outer = np.outer(x[a] - x[b], rest[a] - rest[b])
        covariance[a] += outer
        covariance[b] += outer
    u, _, vt = np.linalg.svd(covariance)
    correction = np.tile(np.eye(3), (len(x), 1, 1))
    # Enforce det(R) = +1 to exclude reflections.
    correction[:, 2, 2] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def objective(x, rest, edges, rotation):
    return sum(
        0.25
        * (
            np.linalg.norm(x[a] - x[b] - rotation[a] @ (rest[a] - rest[b])) ** 2
            + np.linalg.norm(x[a] - x[b] - rotation[b] @ (rest[a] - rest[b])) ** 2
        )
        for a, b in edges
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", nargs="?", type=Path, default=Path("output/python/arap")
    )
    args = parser.parse_args()
    n = 9
    rest = np.array(
        [[j / (n - 1), i / (n - 1), 0.0] for i in range(n) for j in range(n)]
    )
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces.extend([[a, a + 1, a + n + 1], [a, a + n + 1, a + n]])
    edges = sorted(
        {tuple(sorted((f[k], f[(k + 1) % 3]))) for f in faces for k in range(3)}
    )
    edge_vertices = np.array(edges)
    edge_map = dof.StencilDofMap(n * n, edges, 3)
    edge_hessian = np.kron([[1.0, -1.0], [-1.0, 1.0]], np.eye(3))
    angle = np.pi / 3
    R = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    rigid = rest @ R.T + [0.3, -0.2, 0.7]
    assert objective(rigid, rest, edges, rotations(rigid, rest, edges)) < 1e-24
    x = rest.copy()
    right = np.arange(n) * n + n - 1
    x[right] = (rest[right] - [1.0, 0.5, 0.0]) @ R.T + [0.8, 0.5, 0.5]
    handles = np.concatenate([np.arange(n) * n, right])
    fixed_dofs = (3 * handles[:, None] + np.arange(3)).ravel()
    # DofReduction stores indices; prescribed values come from the initial x.
    fixed = dof.DofReduction(x.size, fixed_dofs.tolist())
    initial = x.copy()
    rotation = rotations(x, rest, edges)
    first = previous = objective(x, rest, edges, rotation)
    options = solver.NewtonOptions(gradient_tolerance=1e-10)
    for iteration in range(1000):

        def edge_term(ids, local_x, need_gradient, need_hessian):
            local_x = local_x.reshape(len(ids), 2, 3)
            # Both directed contributions, each with weight 1/4.
            i, j = edge_vertices[ids].T
            p = rest[i] - rest[j]
            d = local_x[:, 0] - local_x[:, 1]
            ri = d - np.einsum("bij,bj->bi", rotation[i], p)
            rj = d - np.einsum("bij,bj->bi", rotation[j], p)
            force = 0.5 * (ri + rj)
            g = np.concatenate([force, -force], axis=1) if need_gradient else None
            h = (
                np.broadcast_to(edge_hessian, (len(ids), 6, 6))
                if need_hessian
                else None
            )
            return (
                0.25 * np.sum(ri * ri + rj * rj, axis=1),
                g,
                h.reshape(-1, h.shape[-1]) if h is not None else None,
            )

        # Rotations stay fixed throughout this global solve.
        global_energy = energy.MeshEnergy(edge_map, edge_term)
        result = solver.newton_solve(global_energy, fixed, x.ravel(), options)
        assert result.status == solver.NewtonStatus.Converged
        updated = result.x.reshape(-1, 3)
        # Local step: refit rotations at the new positions.
        rotation = rotations(updated, rest, edges)
        value = objective(updated, rest, edges, rotation)
        assert value <= previous + 1e-12
        change = np.max(np.abs(updated - x))
        x = updated
        previous = value
        if change < 1e-8:
            break
    else:
        raise RuntimeError("ARAP did not converge within 1000 local/global iterations")
    np.testing.assert_array_equal(x[handles], initial[handles])
    assert value < first and np.linalg.norm(x[n * n // 2] - rest[n * n // 2]) > 0.05
    args.output.mkdir(parents=True, exist_ok=True)
    topology = mesh.TriangleTopology(n * n, np.array(faces, dtype=np.int32))
    for name, positions in [("rest", rest), ("handles", initial), ("edited", x)]:
        mesh.write_obj(
            args.output / f"{name}.obj", mesh.TriangleMesh(positions, topology)
        )
    print(
        f"ARAP: {iteration + 1} iterations, energy {first:g} -> {value:g}, max update={change:g}"
    )
    print(f"OBJ meshes: {args.output}")


if __name__ == "__main__":
    main()
