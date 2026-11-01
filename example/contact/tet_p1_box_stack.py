"""Python mirror of tet_p1_box_stack.cpp: press two tet boxes together and solve static equilibrium with contact.

Run from the repository root after building litepgo:
    .venv/bin/python example/contact/tet_p1_box_stack.py
"""

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
        default=REPO / "output/python/tet_p1_box_stack.veg",
        help="deformed VEG mesh; its boundary surface is written beside it as OBJ",
    )
    args = parser.parse_args()
    box = mesh.read_veg(args.input)
    if box.topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty mesh")
    box_vertices = box.topology.num_vertices
    box_nodes = box.positions
    bottom = box_nodes[:, 1].min()
    height = box_nodes[:, 1].max() - bottom
    if not height > 0:
        raise RuntimeError("Example requires a box with nonzero height")

    # Two copies of the box, one body each, the upper one a small gap above the lower.
    gap = 0.04 * height
    # A small lateral offset avoids an exactly symmetric contact bifurcation.
    rest_nodes = np.vstack(
        [box_nodes, box_nodes + [0.013 * height, height + gap, 0.017 * height]]
    )
    rest_positions = rest_nodes.ravel()
    cells = np.vstack([box.cells, box.cells + box_vertices]).astype(np.int32)
    stack = mesh.TetMesh(rest_nodes, cells)
    topology = stack.topology

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
    num_dofs = elastic.num_dofs

    # The lower box's bottom face is clamped; the upper box's top face is pulled down
    # by more than the gap, so the boxes press against each other.
    tolerance, press, stiffness = height * 1e-8, 0.2 * height, 1.0
    clamped_vertices = np.flatnonzero(np.abs(box_nodes[:, 1] - bottom) <= tolerance)
    pulled_vertices = (
        np.flatnonzero(np.abs(box_nodes[:, 1] - bottom - height) <= tolerance)
        + box_vertices
    )
    clamped = (3 * clamped_vertices[:, None] + np.arange(3)).ravel().tolist()
    pulled_dofs = (3 * pulled_vertices[:, None] + np.arange(3)).astype(np.int32)
    targets = np.zeros((len(pulled_vertices), 3))
    targets[:, 1] = -press
    pulling = energy.VertexPullingEnergy(num_dofs, pulled_dofs, targets, stiffness)

    # Contact acts on the boundary of both boxes, whose vertices move with their
    # volume displacement DOFs. The barrier's weight kappa is its stiffness.
    extracted = mesh.extract_surface_with_vertices(stack)
    surface = extracted.mesh
    W = dof.build_vertex_displacement_map(extracted.vertices, num_dofs)
    contact_options = contact.IPCContactOptions(dhat=0.01 * height)
    barrier = contact.IPCContactEnergy(surface, contact_options, contact.SpatialHash())
    kinematics = dof.SurfaceKinematics(surface.positions, W)
    contact_energy = energy.AffineMappedEnergy(
        barrier, kinematics.displacement_map, kinematics.rest_positions.ravel()
    )
    kappa = 1e3

    potential = energy.WeightedEnergySum(
        [(elastic, 1.0), (pulling, 1.0), (contact_energy, kappa)]
    )
    fixed = dof.DofReduction(num_dofs, clamped)
    initial = np.zeros(num_dofs)
    options = solver.NewtonOptions()
    options.gradient_tolerance = 1e-8
    options.max_iterations = 200
    sparse_solver = linalg.AutoSparseSolver()
    result = solver.newton_solve(potential, fixed, initial, options, sparse_solver)
    converged = result.status == solver.NewtonStatus.Converged

    # The boxes touch, within dhat, without crossing.
    surface_positions = kinematics.positions(result.x)
    pairs = barrier.num_active_pairs(surface_positions)
    distance = barrier.min_active_distance(surface_positions)
    touching = pairs > 0 and 0 < distance < contact_options.dhat
    print(
        f"converged={int(converged)}, iterations={result.iterations}, potential={result.energy:.6g}, "
        f"elastic={elastic.eval_value(result.x):.6g}, pulling={pulling.eval_value(result.x):.6g}, "
        f"contact={kappa * contact_energy.eval_value(result.x):.6g}, free residual={result.gradient_norm:.6g}, "
        f"active pairs={pairs}, min distance={distance:.6g}"
    )
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    deformed = mesh.TetMesh((rest_positions + result.x).reshape(-1, 3), topology)
    mesh.write_veg(output, deformed)
    mesh.write_obj(output.with_suffix(".obj"), mesh.extract_surface(deformed))
    return 0 if converged and touching else 1


if __name__ == "__main__":
    sys.exit(main())
