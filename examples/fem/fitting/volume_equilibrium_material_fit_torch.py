"""Fit two material parameters from equilibrium reactions with Torch L-BFGS.

Run with the optional torch dependency installed. The FEM problem includes
both displacement and direct material contributions to reaction derivatives.

Setup: uv pip install torch
Run: uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit_torch.py
Requires a built litepgo Python package. Uses CPU float64 and Torch L-BFGS.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from litepgo import torch as lp_torch
from litepgo import energy, dof, fem, mesh, solver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[3] / "assets/box-tet.veg",
        help="rest tetrahedral VEG mesh",
    )
    args = parser.parse_args()
    # Build the mesh, FEM model and prescribed boundary displacement.
    rest = mesh.read_veg(args.input)
    topology = rest.topology
    positions = rest.positions.ravel()
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedField3D[fem.TetP1Basis]
    ElasticEnergy = fem.VolumetricFemEnergy[
        fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
    ]
    geometry = [XField(mapping.gather(e, positions)) for e in range(topology.num_cells)]
    z = rest.positions[:, 2]
    height = z.max() - z.min()
    if not height > 0:
        raise RuntimeError("Empty height")
    top = np.flatnonzero(np.abs(z - z.max()) < height * 1e-8)
    boundary = np.flatnonzero(
        (np.abs(z - z.min()) < height * 1e-8) | (np.abs(z - z.max()) < height * 1e-8)
    )
    fixed_dofs = (3 * boundary[:, None] + np.arange(3)).ravel()
    if not 0 < fixed_dofs.size < positions.size:
        raise RuntimeError("Calibration requires constrained and free nodes")
    fixed = dof.DofReduction(positions.size, fixed_dofs.tolist())
    solves = 0

    # Solve equilibrium before measuring reactions; free DOFs move with the parameters.
    def reaction(theta, axis, strain, need_jacobian):
        nonlocal solves
        if not np.isfinite(theta).all() or (theta <= 0).any():
            return None
        elastic = ElasticEnergy(
            mapping,
            geometry,
            [fem.IdentityField3D(np.eye(3).ravel())] * topology.num_cells,
            [fem.IdentityField3D(theta)] * topology.num_cells,
        )
        initial = np.zeros(positions.size)
        loaded = 3 * top + axis
        initial[loaded] = strain * height
        options = solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100)
        ForceParameterResponse = fem.VolumetricFemForceMaterialResponse[
            fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
        ]

        def make_energy(parameters):
            nonlocal solves
            solves += 1
            return ElasticEnergy(
                mapping,
                geometry,
                [fem.IdentityField3D(np.eye(3).ravel())] * topology.num_cells,
                [fem.IdentityField3D(parameters)] * topology.num_cells,
            )

        def force_parameter_jacobian(parameters, displacement):
            response = ForceParameterResponse(
                dof.ConstantDofMap(topology, 2),
                mapping,
                geometry,
                displacement,
                [fem.IdentityField3D(np.eye(3).ravel())] * topology.num_cells,
            )
            return response.eval_jacobian(parameters)

        equilibrium = solver.StaticEquilibriumResponse(
            2, fixed, initial, options, make_energy, force_parameter_jacobian
        )
        displacement, jacobian = equilibrium.eval(theta, need_jacobian=need_jacobian)
        if not np.array_equal(displacement[fixed_dofs], initial[fixed_dofs]):
            raise RuntimeError("Prescribed displacement changed")
        force = elastic.eval_gradient(displacement)[loaded].sum()
        if not need_jacobian:
            return force, None
        force_derivative = (
            force_parameter_jacobian(theta, displacement)
            + elastic.eval_hessian(displacement) @ jacobian
        )
        derivative = np.asarray(force_derivative[loaded].sum(axis=0)).ravel()
        if abs(derivative @ theta - force) > 1e-7 * max(1, abs(force)):
            raise RuntimeError("Reaction sensitivity violates stiffness scaling")
        return force, derivative

    def measurements(theta, need_jacobian):
        results = [
            reaction(theta, 0, 0.08, need_jacobian),
            reaction(theta, 2, -0.04, need_jacobian),
        ]
        if any(value is None for value in results):
            return None
        return np.array([value[0] for value in results]), np.array(
            [value[1] for value in results]
        )

    truth = np.array([2.0, 3.0])
    observed = measurements(truth, False)
    if observed is None:
        raise RuntimeError("Observation solve failed")
    observed = observed[0]
    scales = np.maximum(np.abs(observed), 1e-8)

    # Optimize log(theta) so both material parameters stay positive.
    class MaterialResidual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((2, 2)))

        def eval_values(self, log_parameters, need_jacobian):
            with np.errstate(over="ignore"):
                theta = np.exp(log_parameters)
            values = measurements(theta, need_jacobian)
            if values is None:
                return None
            forces, jacobian = values
            r = (forces - observed) / scales
            return (
                (r, jacobian * theta[None, :] / scales[:, None]) if need_jacobian else r
            )

    residual = MaterialResidual()
    log_parameters = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    # L-BFGS may evaluate this closure repeatedly during its line search.
    optimizer = torch.optim.LBFGS(
        [log_parameters],
        max_iter=60,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        r = lp_torch.response(residual, log_parameters)
        loss = 0.5 * r.square().sum()
        # The adapter applies the FEM Jacobian transpose to the upstream gradient.
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = log_parameters.detach().exp().numpy()
    error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    # Validate on a load that was not used to construct the fitting target.
    reference = reaction(truth, 2, -0.07, False)[0]
    predicted = reaction(fitted, 2, -0.07, False)[0]
    held_out = abs(predicted - reference) / abs(reference)
    print(
        f"truth={truth}, fitted={fitted}, parameter error={error:.3g}, "
        f"held-out error={held_out:.3g}, solves={solves}"
    )
    return 0 if error < 1e-4 and held_out < 1e-4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
