"""Differential plastic growth of a clamped shell strip; solve u and write OBJ.

Run from the repository root after building litepgo:
    .venv/bin/python example/fem/plasticity/shell_plastic_field.py
"""

import argparse
from pathlib import Path

import numpy as np

from litepgo import dof, fem, linalg, mesh, solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "output/python/shell_plastic_field",
    )
    args = parser.parse_args()
    nx, ny = 12, 4
    nodes = np.array(
        [[4 * i / nx, j / ny - 0.5, 0.0] for i in range(nx + 1) for j in range(ny + 1)]
    )
    faces = []
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            b = a + ny + 1
            faces.extend([[a, b, b + 1], [a, b + 1, a + 1]])
    topology = mesh.TriangleTopology(len(nodes), np.array(faces, dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(len(nodes), stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(
            mapping.gather(e, nodes.ravel()), stencils.opposite[e]
        )
        for e in range(mapping.num_elements)
    ]
    thickness = [0.2] * mapping.num_elements
    theta = [np.array([2.0, 3.0])] * mapping.num_elements
    # Two vertex columns clamp both position and the direction of the strip.
    fixed_dofs = list(range(3 * 2 * (ny + 1)))
    fixed = dof.DofReduction(mapping.num_dofs, fixed_dofs)
    u = np.zeros(mapping.num_dofs)
    options = solver.NewtonOptions()
    options.gradient_tolerance = 1e-9
    options.max_iterations = 100
    args.output.mkdir(parents=True, exist_ok=True)
    mesh.write_obj(args.output / "rest.obj", mesh.TriangleMesh(nodes, topology))
    for step in range(5):
        alpha = 0.04 * step / 4
        plastic = [
            np.diag([np.exp(alpha * 2 * r.X_local[:3, 1].mean()), 1.0, 1.0])
            for r in rest
        ]
        elastic = fem.ShellFemEnergy[fem.KoiterStVK](
            mapping, rest, thickness, theta, plastic
        )
        result = solver.newton_solve(
            elastic, fixed, u, options, linalg.AutoSparseSolver()
        )
        if result.status != solver.NewtonStatus.Converged:
            raise RuntimeError(f"Static equilibrium failed at step {step}")
        u = result.x
        residual = np.linalg.norm(elastic.eval_gradient(u)[len(fixed_dofs) :])
        max_displacement = np.linalg.norm(u.reshape(-1, 3), axis=1).max()
        print(
            f"step={step}, energy={result.energy:.12g}, free_residual={residual:.12g}, "
            f"max_displacement={max_displacement:.12g}"
        )
        if residual > 1e-8 or np.max(np.abs(u[fixed_dofs])) > 1e-14:
            raise RuntimeError("Equilibrium or clamp check failed")
        if step == 0 and max_displacement > 1e-12:
            raise RuntimeError("Identity plastic field changed the rest pose")
    if max_displacement < 0.01:
        raise RuntimeError("Plastic field did not deform the strip")
    mesh.write_obj(
        args.output / "equilibrium.obj",
        mesh.TriangleMesh(nodes + u.reshape(-1, 3), topology),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
