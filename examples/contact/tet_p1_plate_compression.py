"""Compress two tetrahedral boxes between penalty plates with IPC self-contact."""
from litepgo.solver import AbsoluteConvergence

from litepgo.energy import AffineResponse
import argparse
import sys
from pathlib import Path

import numpy as np

from litepgo import contact, dof, energy, fem, linalg, mesh, solver

REPO = Path(__file__).resolve().parents[2]


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
        default=REPO / "output/python/tet_p1_plate_compression.veg",
        help="deformed VEG mesh; its boundary surface is written beside it as OBJ",
    )
    args = parser.parse_args()
    box = mesh.read_veg(args.input)
    if box.topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty mesh")
    box_vertices = box.topology.num_vertices
    box_nodes = box.positions
    bottom = box_nodes[:, 2].min()
    height = box_nodes[:, 2].max() - bottom
    if not height > 0:
        raise RuntimeError("Example requires a box with nonzero height")

    # Two copies of the box, one body each, the upper one a small gap above the lower.
    gap = 0.04 * height
    # A small lateral offset avoids an exactly symmetric contact bifurcation.
    rest_nodes = np.vstack(
        [box_nodes, box_nodes + [0.013 * height, 0.017 * height, height + gap]]
    )
    rest_positions = rest_nodes.ravel()
    cells = np.vstack([box.cells, box.cells + box_vertices]).astype(np.int32)
    stack = mesh.TetMesh(rest_nodes, cells)
    topology = stack.topology

    rest_position_map = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedResponse[fem.TetP1Basis]
    rest_position_fields = [
        XField(rest_position_map.gather(e, rest_positions))
        for e in range(topology.num_cells)
    ]
    Fp_field = AffineResponse(np.zeros((9, 3)), np.eye(3).ravel(order="F"))
    theta_field = AffineResponse(np.zeros((2, 3)), np.array([0.5, 0.75]))
    elastic = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature](fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
        rest_position_map,
        rest_position_fields,
        [Fp_field] * topology.num_cells,
        [theta_field] * topology.num_cells,
    )
    num_dofs = elastic.num_dofs

    # Pin the lateral DOFs on the outer faces to remove rigid modes.
    tolerance = height * 1e-8
    lower = np.flatnonzero(np.abs(box_nodes[:, 2] - bottom) <= tolerance)
    upper = (
        np.flatnonzero(np.abs(box_nodes[:, 2] - bottom - height) <= tolerance)
        + box_vertices
    )
    clamped = (
        (3 * np.concatenate([lower, upper])[:, None] + np.arange(2)).ravel().tolist()
    )

    # Contact acts on the boundary of both boxes, whose vertices move with their
    # volume displacement DOFs. The barrier's weight kappa is its stiffness.
    extracted = mesh.extract_surface_with_vertices(stack)
    surface = extracted.mesh
    W = dof.build_vertex_displacement_map(extracted.vertices, num_dofs)
    contact_options = contact.IPCContactOptions(dhat=0.004 * height)
    barrier = contact.IPCContactEnergy(surface, contact_options, contact.SpatialHash())
    kinematics = dof.SurfaceKinematics(surface.positions, W)
    rest_surface = kinematics.rest_positions.ravel()
    n = surface.topology.num_vertices
    bottom_plate = contact.PlanePenaltyEnergy(
        n, axis=2, height=bottom + 0.2 * height, stiffness=20.0
    )
    top_plate = contact.PlanePenaltyEnergy(
        n,
        axis=2,
        height=bottom + 1.8 * height + gap,
        stiffness=20.0,
        positive_side=False,
    )
    kappa = 1e7
    potential = energy.WeightedSumEnergy(
        [
            (elastic, 1.0),
            (energy.AffineMappedEnergy(barrier, W, rest_surface), kappa),
            (energy.AffineMappedEnergy(bottom_plate, W, rest_surface), 1.0),
            (energy.AffineMappedEnergy(top_plate, W, rest_surface), 1.0),
        ]
    )
    fixed = dof.DofReduction(num_dofs, clamped)
    initial = np.zeros(num_dofs)
    options = solver.NewtonOptions()
    options.max_iterations = 200
    sparse_solver = linalg.AutoSparseSolver()
    result = solver.newton_solve(potential, fixed, initial, options, sparse_solver, convergence=AbsoluteConvergence(1e-8))
    converged = result.status == solver.NewtonStatus.Converged

    surface_positions = kinematics.positions(result.x)
    pairs = barrier.num_active_pairs(surface_positions)
    distance = barrier.min_active_distance(surface_positions)
    lower_energy = bottom_plate.evaluate(surface_positions).value
    upper_energy = top_plate.evaluate(surface_positions).value
    print(
        f"converged={int(converged)}, iterations={result.iterations}, potential={result.energy:.6g}, "
        f"bottom penalty={lower_energy:.6g}, top penalty={upper_energy:.6g}, "
        f"active pairs={pairs}, min distance={distance:.6g}"
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    deformed = mesh.TetMesh((rest_positions + result.x).reshape(-1, 3), topology)
    mesh.write_veg(output, deformed)
    mesh.write_obj(output.with_suffix(".obj"), mesh.extract_surface(deformed))
    plates_engaged = lower_energy > 0 and upper_energy > 0
    touching = pairs > 0 and 0 < distance < contact_options.dhat
    return 0 if converged and plates_engaged and touching else 1


if __name__ == "__main__":
    sys.exit(main())
