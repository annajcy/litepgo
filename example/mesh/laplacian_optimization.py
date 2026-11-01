"""Boundary-fixed smoothing and differential-coordinate mesh editing.

Run from the repository root after building litepgo:
    .venv/bin/python example/mesh/laplacian_optimization.py
"""

import argparse
from pathlib import Path

import numpy as np

from litepgo import dof, energy, mesh, solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("output/python/laplacian_optimization"),
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
    topology = mesh.TriangleTopology(n * n, np.array(faces, dtype=np.int32))
    edges = sorted(
        {
            tuple(sorted((face[k], face[(k + 1) % 3])))
            for face in faces
            for k in range(3)
        }
    )
    edge_map = dof.StencilDofMap(n * n, edges, 3)
    edge_hessian = np.kron([[1.0, -1.0], [-1.0, 1.0]], np.eye(3))

    def edge_term(ids, x, need_gradient, need_hessian):
        x = x.reshape(len(ids), 2, 3)
        # Each row is one undirected edge: 1/2 ||x_i - x_j||^2.
        d = x[:, 0] - x[:, 1]
        g = np.concatenate([d, -d], axis=1) if need_gradient else None
        h = np.broadcast_to(edge_hessian, (len(ids), 6, 6)) if need_hessian else None
        return (
            0.5 * np.sum(d * d, axis=1),
            g,
            h.reshape(-1, h.shape[-1]) if h is not None else None,
        )

    neighbors = [[] for _ in rest]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    rings = [[i, *adjacent] for i, adjacent in enumerate(neighbors)]
    ring_map = dof.StencilDofMap(n * n, rings, 3)

    def ring_term(ids, x, need_gradient, need_hessian):
        x = x.reshape(len(ids), -1, 3)
        # Each batch groups equal-length one-rings; ids retain original vertex IDs.
        k = x.shape[1]
        weights = np.array([k - 1, *([-1.0] * (k - 1))])
        B = np.kron(weights[None, :], np.eye(3))
        p = rest[np.array([rings[i] for i in ids])]
        r = np.einsum("k,bkd->bd", weights, x - p)
        g = r @ B if need_gradient else None
        h = np.broadcast_to(B.T @ B, (len(ids), 3 * k, 3 * k)) if need_hessian else None
        return (
            0.5 * np.sum(r * r, axis=1),
            g,
            h.reshape(-1, h.shape[-1]) if h is not None else None,
        )

    boundary = [
        i * n + j
        for i in range(n)
        for j in range(n)
        if i in (0, n - 1) or j in (0, n - 1)
    ]
    noisy = rest.copy()
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            noisy[i * n + j, 2] = 0.15 * np.sin(0.7 * i) * np.sin(1.1 * j)
    options = solver.NewtonOptions(gradient_tolerance=1e-10)
    smooth_energy = energy.MeshEnergy(edge_map, edge_term)
    boundary_dofs = (3 * np.array(boundary)[:, None] + np.arange(3)).ravel()
    # Solve only the free coordinates; prescribed coordinates retain their initial values.
    smooth = solver.newton_solve(
        smooth_energy,
        dof.DofReduction(3 * n * n, boundary_dofs.tolist()),
        noisy.ravel(),
        options,
    )
    assert smooth.status == solver.NewtonStatus.Converged
    np.testing.assert_allclose(smooth.x, rest.ravel(), atol=1e-9, rtol=0)
    np.testing.assert_array_equal(smooth.x[boundary_dofs], noisy.ravel()[boundary_dofs])
    assert smooth.energy < smooth_energy.eval_value(noisy.ravel())

    # MeshEnergy gathers each stencil and assembles its local derivatives.
    edit_energy = energy.MeshEnergy(ring_map, ring_term)
    handles = [i * n + j for i in range(n) for j in (0, n - 1)]
    handle_dofs = (3 * np.array(handles)[:, None] + np.arange(3)).ravel()
    initial = rest.copy()
    initial[np.arange(n) * n + n - 1, 2] = 0.5
    edited = solver.newton_solve(
        edit_energy,
        dof.DofReduction(3 * n * n, handle_dofs.tolist()),
        initial.ravel(),
        options,
    )
    assert edited.status == solver.NewtonStatus.Converged
    np.testing.assert_array_equal(edited.x[handle_dofs], initial.ravel()[handle_dofs])
    np.testing.assert_allclose(
        edited.x.reshape(-1, 3)[:, :2], rest[:, :2], atol=1e-9, rtol=0
    )
    assert edited.x.reshape(-1, 3)[n * n // 2, 2] > 0.1
    before = np.sqrt(2.0 * edit_energy.eval_value(initial.ravel()))
    after = np.sqrt(2.0 * edit_energy.eval_value(edited.x))
    assert after < before
    args.output.mkdir(parents=True, exist_ok=True)
    for name, positions in [
        ("rest", rest),
        ("noisy", noisy),
        ("smoothed", smooth.x.reshape(-1, 3)),
        ("handles", initial),
        ("edited", edited.x.reshape(-1, 3)),
    ]:
        mesh.write_obj(
            args.output / f"{name}.obj", mesh.TriangleMesh(positions, topology)
        )
    print(
        f"smoothing: energy={smooth.energy:g}, free_gradient={smooth.gradient_norm:g}"
    )
    print(
        f"editing: differential residual {before:g} -> {after:g}, free_gradient={edited.gradient_norm:g}"
    )
    print(f"OBJ meshes: {args.output}")


if __name__ == "__main__":
    main()
