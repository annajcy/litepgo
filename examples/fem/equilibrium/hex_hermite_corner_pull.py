"""Python mirror of hex_hermite_corner_pull.cpp: pull a Hermite hex box's corners and solve static equilibrium.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/equilibrium/hex_hermite_corner_pull.py
"""
from litepgo.solver import AbsoluteConvergence

from litepgo.energy import AffineResponse
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
        default=REPO / "assets/box-hex.veg",
        help="rest VEG mesh",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=REPO / "output/python/hex_hermite_corner_pull.veg",
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
    displacement_map = dof.VertexDofMap(topology, 24)
    XField = fem.InterpolatedResponse[fem.HexQ1Basis]
    rest_position_fields = [
        XField(rest_position_map.gather(e, rest_positions))
        for e in range(topology.num_cells)
    ]
    Fp_field = AffineResponse(np.zeros((9, 3)), np.eye(3).ravel(order="F"))
    theta_field = AffineResponse(np.zeros((2, 3)), np.array([2.0, 3.0]))
    elastic = fem.VolumetricFemEnergy[fem.HexHermiteBasis, fem.GaussLegendre3D[4]](fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
        displacement_map,
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
    pulled_dofs = (24 * corners[:, None] + np.arange(3)).astype(
        np.int32
    )  # displacement slots only
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
    sparse_solver = linalg.AutoSparseSolver()
    result = solver.newton_solve(
        potential, unconstrained, initial, options, sparse_solver,
        convergence=AbsoluteConvergence(1e-8),
    )
    converged = result.status == solver.NewtonStatus.Converged
    corner_gap = np.linalg.norm(
        result.x.reshape(-1, 24)[corners, :3] - targets, axis=1
    ).max()
    print(
        f"converged={int(converged)}, iterations={result.iterations}, potential={result.energy:.6g}, "
        f"elastic={elastic.evaluate(result.x).value:.6g}, pulling={pulling.evaluate(result.x).value:.6g}, "
        f"free residual={result.gradient_norm:.6g}, max corner-anchor distance={corner_gap:.6g}"
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    deformed = mesh.HexMesh(rest_nodes + result.x.reshape(-1, 24)[:, :3], topology)
    mesh.write_veg(output, deformed)
    mesh.write_obj(output.with_suffix(".obj"), mesh.extract_surface(deformed))
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
