"""Python mirror of hex_hermite_plastic_twist.cpp: release and clamp a Hermite hex bar with a plastic twist Fp.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/plasticity/hex_hermite_plastic_twist.py
"""
from litepgo.solver import AbsoluteConvergence

from litepgo.energy import AffineResponse
import argparse
import sys
from pathlib import Path

import numpy as np

from litepgo import dof, fem, linalg, mesh, solver

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
        default=REPO / "output/python/plastic_twist/hex_hermite_plastic_twist.veg",
        help="base VEG path; <stem>_free and <stem>_clamped VEG and surface OBJ files are written beside it",
    )
    args = parser.parse_args()
    rest_mesh = mesh.read_veg(args.input)
    topology = rest_mesh.topology
    rest_positions = rest_mesh.positions.ravel()
    rest_nodes = rest_mesh.positions
    if topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty mesh")

    # Plastic twist: u_p is the displacement a bar twisted past yield keeps once released, here a twist
    # about the vertical center axis growing linearly to 90 degrees at the top. Fp = I + du_p/dX, the
    # gradient of X + u_p, is a rotation times a shear (det Fp = 1), not a rotation. Released, the bar
    # takes the shape X + u_p stress-free; held at both ends, it cannot and keeps residual stress.
    # u_p uses the Hermite basis, so it is C1 like the displacement. Its 24 physical DOFs per vertex are
    # the twist's value and derivatives u, u_X, u_Y, u_Z, u_XY, u_XZ, u_YZ, u_XYZ; for each Z it is
    # linear in X and Y, so u_XY and u_XYZ vanish.
    lower = rest_nodes.min(axis=0)
    upper = rest_nodes.max(axis=0)
    center = (lower + upper) / 2
    height = upper[2] - lower[2]
    if not height > 0:
        raise RuntimeError("Example requires nonzero height")
    rate = np.pi / 2 / height  # twist angle per unit height
    r = (rest_nodes - center)[:, :2]
    angle = rate * (rest_nodes[:, 2] - lower[2])
    c, s = np.cos(angle), np.sin(angle)
    p = np.column_stack(
        [c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1]]
    )  # r rotated by angle
    slots = np.zeros((topology.num_vertices, 8, 3))
    slots[:, 0, :2] = p - r
    slots[:, 1, :2] = np.column_stack([c - 1, s])
    slots[:, 2, :2] = np.column_stack([-s, c - 1])
    slots[:, 3, :2] = rate * np.column_stack([-p[:, 1], p[:, 0]])
    slots[:, 5, :2] = rate * np.column_stack([-s, c])
    slots[:, 6, :2] = rate * np.column_stack([-c, -s])
    plastic_dofs = slots.ravel()

    # DeformationGradientResponse owns its fixed displacement and geometry responses. BasisDofTransform
    # converts each element's physical Hermite DOFs to basis coefficients, as the energy does for u.
    rest_position_map = dof.VertexDofMap(topology, 3)
    displacement_map = dof.VertexDofMap(topology, 24)
    XField = fem.InterpolatedResponse[fem.HexQ1Basis]
    PlasticField = fem.InterpolatedResponse[fem.HexHermiteBasis]
    rest_position_fields = [
        XField(rest_position_map.gather(e, rest_positions))
        for e in range(topology.num_cells)
    ]
    plastic_fields = []
    for e, X in enumerate(rest_position_fields):
        transform = fem.BasisDofTransform[fem.HexHermiteBasis](X)
        plastic_fields.append(
            PlasticField(
                transform.to_coefficients(displacement_map.gather(e, plastic_dofs))
            )
        )
    Fp_fields = [
        fem.DeformationGradientResponse(u_p, X)
        for u_p, X in zip(plastic_fields, rest_position_fields)
    ]
    theta_field = AffineResponse(np.zeros((2, 3)), np.array([2.0, 3.0]))
    elastic = fem.VolumetricFemEnergy[fem.HexHermiteBasis, fem.GaussLegendre3D[4]](fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
        displacement_map,
        rest_position_fields,
        Fp_fields,
        [theta_field] * topology.num_cells,
    )

    tolerance = height * 1e-8
    bottom = np.flatnonzero(np.abs(rest_nodes[:, 2] - lower[2]) <= tolerance)
    top = np.flatnonzero(np.abs(rest_nodes[:, 2] - upper[2]) <= tolerance)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    surface = fem.sample_boundary[fem.HexQ1Basis](topology, rest_position_fields, 4)
    W = fem.build_point_displacement_map[fem.HexHermiteBasis](
        surface.embedding, displacement_map, rest_position_fields
    )

    # free: clamp the bottom face; the bar twists. clamped: also hold the top face in place;
    # the bar cannot twist and keeps residual stress. Clamping a face z = const fixes u and its
    # in-face derivatives u_X, u_Y, u_XY, which determine the displacement on the whole face.
    converged = True
    for clamp_top in (False, True):
        faces = [bottom, top] if clamp_top else [bottom]
        fixed_dofs = [
            24 * v + 3 * slot + c
            for face in faces
            for v in face
            for slot in (0, 1, 2, 4)
            for c in range(3)
        ]
        fixed = dof.DofReduction(elastic.num_dofs, fixed_dofs)
        initial = np.zeros(elastic.num_dofs)
        options = solver.NewtonOptions()
        options.max_iterations = 200
        sparse_solver = linalg.AutoSparseSolver()
        result = solver.newton_solve(elastic, fixed, initial, options, sparse_solver, convergence=AbsoluteConvergence(1e-8))
        converged = converged and result.status == solver.NewtonStatus.Converged
        # The top face's twist about the axis, from its vertices' summed in-plane rotation.
        r_top = (rest_nodes[top] - center)[:, :2]
        x_top = r_top + result.x.reshape(-1, 24)[top, :2]
        cross = (r_top[:, 0] * x_top[:, 1] - r_top[:, 1] * x_top[:, 0]).sum()
        dot = (r_top * x_top).sum()
        print(
            f"{'clamped' if clamp_top else 'free'}: "
            f"converged={int(result.status == solver.NewtonStatus.Converged)}, iterations={result.iterations}, "
            f"elastic={result.energy:.6g}, free residual={result.gradient_norm:.6g}, top twist={np.arctan2(cross, dot) * 180 / np.pi:.6g} deg"
        )
        path = (
            output.parent / f"{output.stem}{'_clamped' if clamp_top else '_free'}.veg"
        )
        deformed = mesh.HexMesh(rest_nodes + result.x.reshape(-1, 24)[:, :3], topology)
        mesh.write_veg(path, deformed)
        if not converged:
            raise RuntimeError("Hermite equilibrium did not converge")
        positions = surface.mesh.positions + (W @ result.x).reshape(-1, 3)
        mesh.write_obj(
            path.with_suffix(".obj"),
            mesh.TriangleMesh(positions, surface.mesh.topology),
        )
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
