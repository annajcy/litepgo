"""Python mirror of tet_p1_corner_pull.cpp: pull a tet box's corners and solve static equilibrium.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/equilibrium/tet_p1_corner_pull.py
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
        default=REPO / "assets/box-tet.veg",
        help="rest VEG mesh",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=REPO / "output/python/tet_p1_corner_pull.veg",
        help="deformed VEG mesh; its boundary surface is written beside it as OBJ",
    )
    args = parser.parse_args()
    rest_mesh = mesh.read_veg(args.input)
    topology = rest_mesh.topology
    rest_positions = rest_mesh.positions.ravel()
    rest_nodes = rest_mesh.positions
    if topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty mesh")
    rest_position_map = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedField3D[fem.TetP1Basis]
    rest_position_fields = [
        XField(rest_position_map.gather(e, rest_positions))
        for e in range(topology.num_cells)
    ]
    Fp_field = fem.IdentityField3D(np.eye(3).ravel(order="F"))
    theta_field = fem.IdentityField3D(np.array([2.0, 3.0]))
    elastic = fem.VolumetricFemEnergy[
        fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
    ](
        rest_position_map,
        rest_position_fields,
        [Fp_field] * topology.num_cells,
        [theta_field] * topology.num_cells,
    )

    # Pull the bounding-box corners toward a copy of the box enlarged about its minimum
    # corner, whose anchor coincides with that corner's rest position.
    lower = rest_nodes.min(axis=0)
    upper = rest_nodes.max(axis=0)
    tolerance = (upper - lower).max() * 1e-8
    if not ((upper - lower) > tolerance).all():
        raise RuntimeError("Example requires nonzero extent along every axis")
    near = (np.abs(rest_nodes - lower) <= tolerance) | (
        np.abs(rest_nodes - upper) <= tolerance
    )
    corners = np.flatnonzero(near.all(axis=1))
    if len(corners) != 8:
        raise RuntimeError("Example requires a vertex at each bounding-box corner")
    stretch, stiffness = 0.3, 1.0
    pulled_dofs = (3 * corners[:, None] + np.arange(3)).astype(np.int32)
    targets = stretch * (rest_nodes[corners] - lower)
    pulling = energy.VertexPullingEnergy(
        elastic.num_dofs, pulled_dofs, targets, stiffness
    )

    # Static equilibrium minimizes Pi(u) = E_elastic(u) + E_pulling(u); the springs
    # remove all rigid motions, so no DOF is fixed.
    potential = energy.WeightedSumEnergy([(elastic, 1.0), (pulling, 1.0)])
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
    deformed = mesh.TetMesh((rest_positions + result.x).reshape(-1, 3), topology)
    mesh.write_veg(output, deformed)
    mesh.write_obj(output.with_suffix(".obj"), mesh.extract_surface(deformed))
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
