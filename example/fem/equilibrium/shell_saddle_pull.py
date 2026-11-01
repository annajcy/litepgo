"""Python mirror of shell_saddle_pull.cpp: pull a sheet's corners into a saddle and solve static equilibrium.

Run from the repository root after building litepgo:
    .venv/bin/python example/fem/equilibrium/shell_saddle_pull.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from litepgo import dof, energy, fem, linalg, mesh, solver

REPO = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=REPO / "assets/shell.obj",
        help="rest OBJ mesh",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=REPO / "output/python/shell_saddle_pull.obj",
        help="deformed OBJ mesh",
    )
    args = parser.parse_args()
    rest_mesh = mesh.read_obj(args.input)
    topology = rest_mesh.topology
    rest_positions = rest_mesh.positions.ravel()
    rest_nodes = rest_mesh.positions
    if topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty mesh")
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    stencil_map = dof.StencilDofMap(topology.num_vertices, stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(
            stencil_map.gather(f, rest_positions), stencils.opposite[f]
        )
        for f in range(stencil_map.num_elements)
    ]

    thickness, Y, nu = 0.1, 100.0, 0.3
    # Plane-stress Lame parameters for a shell.
    theta = np.array([Y * nu / (1 - nu * nu), Y / (2 * (1 + nu))])
    elastic = fem.ShellFemEnergy[fem.KoiterStVK](
        stencil_map,
        rest,
        [thickness] * stencil_map.num_elements,
        [theta] * stencil_map.num_elements,
    )

    # Pull the four corners of the flat xy-plane sheet outward in plane and alternately
    # up and down out of plane, so the anchors span a saddle.
    lower = rest_nodes.min(axis=0)
    upper = rest_nodes.max(axis=0)
    center = (lower + upper) / 2
    size = (upper - lower)[:2].max()
    tolerance = size * 1e-8
    if not ((upper - lower)[:2] > tolerance).all() or upper[2] - lower[2] > tolerance:
        raise RuntimeError("Example requires a flat sheet in the xy plane")
    near = (np.abs(rest_nodes - lower) <= tolerance) | (
        np.abs(rest_nodes - upper) <= tolerance
    )
    corners = np.flatnonzero(near[:, :2].all(axis=1))
    if len(corners) != 4:
        raise RuntimeError("Example requires a vertex at each bounding-box corner")
    stretch, lift, stiffness = 0.1, 0.2 * size, 10.0
    pulled_dofs = (3 * corners[:, None] + np.arange(3)).astype(np.int32)
    offset = rest_nodes[corners] - center
    targets = np.column_stack(
        [
            stretch * offset[:, 0],
            stretch * offset[:, 1],
            np.where(offset[:, 0] * offset[:, 1] > 0, lift, -lift),
        ]
    )
    pulling = energy.VertexPullingEnergy(
        elastic.num_dofs, pulled_dofs, targets, stiffness
    )

    # Static equilibrium minimizes Pi(u) = E_elastic(u) + E_pulling(u); the springs
    # remove all rigid motions, so no DOF is fixed.
    potential = energy.WeightedEnergySum([(elastic, 1.0), (pulling, 1.0)])
    unconstrained = dof.DofReduction(elastic.num_dofs, [])
    initial = np.zeros(elastic.num_dofs)
    options = solver.NewtonOptions()
    options.gradient_tolerance = 1e-8
    sparse_solver = linalg.AutoSparseSolver()
    result = solver.newton_solve(
        potential, unconstrained, initial, options, sparse_solver
    )
    converged = result.status == solver.NewtonStatus.Converged
    corner_gap = np.linalg.norm(
        result.x.reshape(-1, 3)[corners] - targets, axis=1
    ).max()
    print(
        f"converged={int(converged)}, iterations={result.iterations}, potential={result.energy:.6g}, "
        f"elastic={elastic.eval_value(result.x):.6g}, pulling={pulling.eval_value(result.x):.6g}, "
        f"free residual={result.gradient_norm:.6g}, max corner-anchor distance={corner_gap:.6g}"
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_obj(
        output, mesh.TriangleMesh((rest_positions + result.x).reshape(-1, 3), topology)
    )
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
