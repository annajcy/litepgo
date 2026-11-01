"""Python mirror of hex_q1_plastic_twist.cpp: release and clamp a hex bar with a plastic twist Fp.

Run from the repository root after building litepgo:
    .venv/bin/python example/fem/plasticity/hex_q1_plastic_twist.py
"""

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
        default=REPO / "output/python/plastic_twist/hex_q1_plastic_twist.veg",
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
    lower = rest_nodes.min(axis=0)
    upper = rest_nodes.max(axis=0)
    center = (lower + upper) / 2
    height = upper[2] - lower[2]
    if not height > 0:
        raise RuntimeError("Example requires nonzero height")
    r = rest_nodes - center
    angle = np.pi / 2 * (rest_nodes[:, 2] - lower[2]) / height
    c, s = np.cos(angle), np.sin(angle)
    plastic_displacement = np.column_stack(
        [
            c * r[:, 0] - s * r[:, 1] - r[:, 0],
            s * r[:, 0] + c * r[:, 1] - r[:, 1],
            np.zeros(topology.num_vertices),
        ]
    ).ravel()

    # DeformationGradientField3D(u_p, X) borrows both fields and keeps them alive.
    rest_position_map = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedField3D[fem.HexQ1Basis]
    rest_position_fields = [
        XField(rest_position_map.gather(e, rest_positions))
        for e in range(topology.num_cells)
    ]
    plastic_fields = [
        XField(rest_position_map.gather(e, plastic_displacement))
        for e in range(topology.num_cells)
    ]
    Fp_fields = [
        fem.DeformationGradientField3D(u_p, X)
        for u_p, X in zip(plastic_fields, rest_position_fields)
    ]
    theta_field = fem.IdentityField3D(np.array([2.0, 3.0]))
    elastic = fem.VolumetricFemEnergy[
        fem.NeoHookean, fem.HexQ1Basis, fem.GaussLegendre3D[2]
    ](
        rest_position_map,
        rest_position_fields,
        Fp_fields,
        [theta_field] * topology.num_cells,
    )

    tolerance = height * 1e-8
    bottom = np.flatnonzero(np.abs(rest_nodes[:, 2] - lower[2]) <= tolerance)
    top = np.flatnonzero(np.abs(rest_nodes[:, 2] - upper[2]) <= tolerance)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    # free: clamp the bottom face; the bar twists. clamped: also hold the top face in place;
    # the bar cannot twist and keeps residual stress.
    converged = True
    for clamp_top in (False, True):
        faces = [bottom, top] if clamp_top else [bottom]
        fixed_dofs = [3 * v + c for face in faces for v in face for c in range(3)]
        fixed = dof.DofReduction(elastic.num_dofs, fixed_dofs)
        initial = np.zeros(elastic.num_dofs)
        options = solver.NewtonOptions()
        options.gradient_tolerance = 1e-8
        options.max_iterations = 200
        sparse_solver = linalg.AutoSparseSolver()
        result = solver.newton_solve(elastic, fixed, initial, options, sparse_solver)
        converged = converged and result.status == solver.NewtonStatus.Converged
        # The top face's twist about the axis, from its vertices' summed in-plane rotation.
        r_top = (rest_nodes[top] - center)[:, :2]
        x_top = r_top + result.x.reshape(-1, 3)[top, :2]
        cross = (r_top[:, 0] * x_top[:, 1] - r_top[:, 1] * x_top[:, 0]).sum()
        dot = (r_top * x_top).sum()
        print(
            f"{'clamped' if clamp_top else 'free'}: "
            f"converged={int(result.status == solver.NewtonStatus.Converged)}, iterations={result.iterations}, "
            f"elastic={result.energy:.6g}, top twist={np.arctan2(cross, dot) * 180 / np.pi:.6g} deg"
        )
        path = (
            output.parent / f"{output.stem}{'_clamped' if clamp_top else '_free'}.veg"
        )
        deformed = mesh.HexMesh((rest_positions + result.x).reshape(-1, 3), topology)
        mesh.write_veg(path, deformed)
        mesh.write_obj(path.with_suffix(".obj"), mesh.extract_surface(deformed))
    return 0 if converged else 1


if __name__ == "__main__":
    sys.exit(main())
