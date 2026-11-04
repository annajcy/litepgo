"""Fit volume material parameters from reactions using analytic equilibrium sensitivities.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit.py
"""

from pathlib import Path

import numpy as np

from litepgo import energy, dof, fem, mesh, solver


def make_problem():
    rest = mesh.read_veg(Path(__file__).resolve().parents[3] / "assets/box-tet.veg")
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

        class MaterialModel(energy.ParametricEnergy):
            def __init__(self):
                prototype = self.potential(theta)
                super().__init__(
                    prototype.hessian_pattern, np.ones((prototype.num_dofs, 2))
                )

            def potential(self, p):
                return (make_energy)(p)

            def eval_state_values(self, u, p, need_gradient, hessian_kind):
                potential = self.potential(p)
                potential.update_pattern(u)
                return potential.eval(u, gradient=need_gradient, hessian=hessian_kind)

            def eval_parameter_jacobian(self, u, p):
                return (force_parameter_jacobian)(p, u)

            def eval_feasible_step(self, u, p, du):
                return self.potential(p).max_feasible_step(u, du)

        equilibrium = solver.StaticEquilibriumResponse(
            MaterialModel(),
            fixed,
            initial,
            options,
        )
        solves += 1
        displacement, J = equilibrium.eval(theta, need_jacobian=need_jacobian)
        if not np.array_equal(displacement[fixed_dofs], initial[fixed_dofs]):
            raise RuntimeError("Prescribed displacement changed")
        force = elastic.eval_gradient(displacement)[loaded].sum()
        if not need_jacobian:
            return force, None
        force_derivative = (
            force_parameter_jacobian(theta, displacement)
            + elastic.eval_hessian(displacement) @ J
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

    class MaterialResidual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((2, 2)))

        def eval_values(self, q, need_jacobian):
            with np.errstate(over="ignore"):
                theta = np.exp(q)
            values = measurements(theta, need_jacobian)
            if values is None:
                return None
            forces, J = values
            r = (forces - observed) / scales
            return (r, J * theta[None, :] / scales[:, None]) if need_jacobian else r

    residual = MaterialResidual()
    return residual, truth, reaction, lambda: solves


def main():
    residual, truth, reaction, solve_count = make_problem()
    # Fit the observation residual using its analytic Jacobian.
    result = solver.gauss_newton_solve(
        residual, np.zeros(2), solver.GaussNewtonOptions(gradient_tolerance=1e-8)
    )
    fitted = np.exp(result.x)
    r = residual.eval_value(result.x)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    reference = reaction(truth, 2, -0.07, False)
    prediction = reaction(fitted, 2, -0.07, False)
    if reference is None or prediction is None or r is None:
        raise RuntimeError("Validation solve failed")
    prediction_error = abs(prediction[0] - reference[0]) / abs(reference[0])
    print(
        f"truth={truth[0]:.6g} {truth[1]:.6g}, fitted={fitted[0]:.6g} {fitted[1]:.6g}, "
        f"parameter error={parameter_error:.6g}, residual={np.linalg.norm(r):.6g}, "
        f"held-out error={prediction_error:.6g}, forward solves={solve_count()}"
    )
    return (
        0
        if (
            result.status == solver.GaussNewtonStatus.Converged
            and parameter_error < 1e-4
            and np.linalg.norm(r) < 1e-7
            and prediction_error < 1e-4
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
