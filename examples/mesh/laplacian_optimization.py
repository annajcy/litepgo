"""Boundary-fixed smoothing and differential-coordinate mesh editing.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/mesh/laplacian_optimization.py
"""

import argparse
from pathlib import Path

import numpy as np

from litepgo import dof, energy, mesh, solver

from volume_mesh import end_faces, read_volume


class EdgeSmoothingEnergy(energy.MeshEnergy):
    def __init__(self, mapping, rest_edge_vectors):
        super().__init__(mapping)
        self.rest_edge_vectors = rest_edge_vectors
        self.edge_hessian = np.kron([[1.0, -1.0], [-1.0, 1.0]], np.eye(3))

    def eval_batch(self, ids, x, need_gradient, need_hessian):
        x = x.reshape(len(ids), 2, 3)
        # The volume mode smooths displacement differences across edges.
        d = x[:, 0] - x[:, 1]
        if self.rest_edge_vectors is not None:
            d -= self.rest_edge_vectors[ids]
        g = np.concatenate([d, -d], axis=1) if need_gradient else None
        h = (
            np.broadcast_to(self.edge_hessian, (len(ids), 6, 6))
            if need_hessian
            else None
        )
        return (
            0.5 * np.sum(d * d, axis=1),
            g,
            h.reshape(-1, h.shape[-1]) if h is not None else None,
        )


class DifferentialCoordinateEnergy(energy.MeshEnergy):
    def __init__(self, mapping, rest, rings):
        super().__init__(mapping)
        self.rest = rest
        self.rings = rings

    def eval_batch(self, ids, x, need_gradient, need_hessian):
        x = x.reshape(len(ids), -1, 3)
        # Each batch groups equal-length one-rings; ids retain original vertex IDs.
        k = x.shape[1]
        weights = np.array([k - 1, *([-1.0] * (k - 1))])
        B = np.kron(weights[None, :], np.eye(3))
        p = self.rest[np.array([self.rings[i] for i in ids])]
        r = np.einsum("k,bkd->bd", weights, x - p)
        g = r @ B if need_gradient else None
        h = np.broadcast_to(B.T @ B, (len(ids), 3 * k, 3 * k)) if need_hessian else None
        return (
            0.5 * np.sum(r * r, axis=1),
            g,
            h.reshape(-1, h.shape[-1]) if h is not None else None,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path("output/python/laplacian_optimization"),
    )
    parser.add_argument(
        "--volume", type=Path, help="axis-aligned tetrahedral box VEG mesh"
    )
    args = parser.parse_args()
    volume = None
    if args.volume:
        volume, edges = read_volume(args.volume)
        rest = volume.positions
    else:
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
    edge_map = dof.StencilDofMap(len(rest), edges, 3)
    rest_edge_vectors = None
    if volume is not None:
        starts, ends = np.asarray(edges).T
        rest_edge_vectors = rest[starts] - rest[ends]

    neighbors = [[] for _ in rest]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    rings = [[i, *adjacent] for i, adjacent in enumerate(neighbors)]
    ring_map = dof.StencilDofMap(len(rest), rings, 3)

    if volume is not None:
        outer = np.asarray(mesh.extract_surface_with_vertices(volume).vertices)
        span = np.ptp(rest, axis=0)
        normalized = (rest - rest.min(axis=0)) / span
        # Keep the sides and bottom fixed while smoothing a rippled top.
        top = (normalized[:, 2] >= 1 - 1e-8) & (
            (normalized[:, :2] > 1e-8) & (normalized[:, :2] < 1 - 1e-8)
        ).all(axis=1)
        if not top.any():
            parser.error("volume smoothing requires free vertices inside the top face")
        boundary = outer[~top[outer]]
        noisy = rest.copy()
        x, y, z = normalized.T
        envelope = np.sin(np.pi * x) * np.sin(np.pi * y)
        free = np.setdiff1d(np.arange(len(rest)), boundary)
        ripple = (
            0.16
            * span[2]
            * z
            * envelope
            * (1 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y))
        )
        noisy[free, 2] += ripple[free]
    else:
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
    smooth_energy = EdgeSmoothingEnergy(edge_map, rest_edge_vectors)
    boundary_dofs = (3 * np.array(boundary)[:, None] + np.arange(3)).ravel()
    # Solve only the free coordinates; prescribed coordinates retain their initial values.
    smooth = solver.newton_solve(
        smooth_energy,
        dof.DofReduction(rest.size, boundary_dofs.tolist()),
        noisy.ravel(),
        options,
    )
    assert smooth.status == solver.NewtonStatus.Converged
    np.testing.assert_allclose(smooth.x, rest.ravel(), atol=1e-9, rtol=0)
    np.testing.assert_array_equal(smooth.x[boundary_dofs], noisy.ravel()[boundary_dofs])
    assert smooth.energy < smooth_energy.evaluate(noisy.ravel()).value

    # MeshEnergy gathers each stencil and assembles its local derivatives.
    edit_energy = DifferentialCoordinateEnergy(ring_map, rest, rings)
    left, right = end_faces(rest)
    handles = np.concatenate([left, right])
    handle_dofs = (3 * np.array(handles)[:, None] + np.arange(3)).ravel()
    initial = rest.copy()
    width = np.ptp(rest[:, 0])
    lift = 0.5 * width
    if volume is None:
        initial[right, 2] += lift
    else:
        initial[:, 2] += lift * (rest[:, 0] - rest[:, 0].min()) / width
    edited = solver.newton_solve(
        edit_energy,
        dof.DofReduction(rest.size, handle_dofs.tolist()),
        initial.ravel(),
        options,
    )
    assert edited.status == solver.NewtonStatus.Converged
    np.testing.assert_array_equal(edited.x[handle_dofs], initial.ravel()[handle_dofs])
    np.testing.assert_allclose(
        edited.x.reshape(-1, 3)[:, :2], rest[:, :2], atol=1e-9, rtol=0
    )
    free = np.setdiff1d(np.arange(len(rest)), handles)
    assert np.max(edited.x[3 * free + 2] - rest[free, 2]) > 0.1 * width
    before = np.sqrt(2.0 * edit_energy.evaluate(initial.ravel()).value)
    after = np.sqrt(2.0 * edit_energy.evaluate(edited.x).value)
    assert after < before
    args.output.mkdir(parents=True, exist_ok=True)
    for name, positions in [
        ("rest", rest),
        ("noisy", noisy),
        ("smoothed", smooth.x.reshape(-1, 3)),
        ("handles", initial),
        ("edited", edited.x.reshape(-1, 3)),
    ]:
        if volume is None:
            surface = mesh.TriangleMesh(positions, topology)
        else:
            deformed = mesh.TetMesh(positions, volume.topology)
            mesh.write_veg(args.output / f"{name}.veg", deformed)
            surface = mesh.extract_surface(deformed)
        mesh.write_obj(args.output / f"{name}.obj", surface)
    print(
        f"smoothing: energy={smooth.energy:g}, free_gradient={smooth.gradient_norm:g}"
    )
    print(
        f"editing: differential residual {before:g} -> {after:g}, free_gradient={edited.gradient_norm:g}"
    )
    print(f"OBJ meshes: {args.output}")


if __name__ == "__main__":
    main()
