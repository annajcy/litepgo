"""Python mirror of material_fit.cpp, using analytic equilibrium sensitivities."""

from pathlib import Path

import numpy as np

from litepgo import energy, dof, fem, linalg, mesh, solver


def main():
    rest = mesh.read_veg(Path(__file__).resolve().parents[2] / "assets/box-tet.veg")
    topology = rest.topology
    positions = rest.positions.ravel()
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedField3D[fem.TetP1Basis]
    Elastic = fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
    geometry = [XField(mapping.gather(e, positions)) for e in range(topology.num_cells)]
    z = rest.positions[:, 2]
    height = z.max() - z.min()
    if not height > 0:
        raise RuntimeError("Empty height")
    top = np.flatnonzero(np.abs(z - z.max()) < height * 1e-8)
    boundary = np.flatnonzero((np.abs(z - z.min()) < height * 1e-8) | (np.abs(z - z.max()) < height * 1e-8))
    fixed_dofs = (3 * boundary[:, None] + np.arange(3)).ravel()
    if not 0 < fixed_dofs.size < positions.size:
        raise RuntimeError("Calibration requires constrained and free nodes")
    fixed = dof.DofReduction(positions.size, fixed_dofs.tolist())
    solves = 0

    def reaction(theta, axis, strain, need_jacobian):
        nonlocal solves
        if not np.isfinite(theta).all() or (theta <= 0).any():
            return None
        elastic = Elastic(mapping, geometry,
                          [fem.IdentityField3D(np.eye(3).ravel())] * topology.num_cells,
                          [fem.IdentityField3D(theta)] * topology.num_cells)
        initial = np.zeros(positions.size)
        loaded = 3 * top + axis
        initial[loaded] = strain * height
        linear_solver = linalg.AutoSparseSolver()
        options = solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100)
        solves += 1
        result = solver.newton_solve(elastic, fixed, initial, options, linear_solver)
        if result.status != solver.NewtonStatus.Converged:
            return None
        if not np.array_equal(result.x[fixed_dofs], initial[fixed_dofs]):
            raise RuntimeError("Prescribed displacement changed")
        force = elastic.eval_gradient(result.x)[loaded].sum()
        if not need_jacobian:
            return force, None

        # Fixed-state partial derivative; the sensitivity solve below accounts for du/dtheta.
        Force = fem.VolumetricFemForceMaterialResponse[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
        force_response = Force(dof.ConstantDofMap(topology, 2), mapping, geometry, result.x,
                               [fem.IdentityField3D(np.eye(3).ravel())] * topology.num_cells)
        B = force_energy.eval_jacobian(theta).toarray()
        K = elastic.eval_hessian(result.x)  # Python eval_hessian returns the exact Hessian.
        if not linear_solver.compute(fixed.reduce(K)):
            raise RuntimeError("Sensitivity factorization failed")
        derivative = np.zeros(2)
        for j in range(2):
            sensitivity = np.zeros(positions.size)
            fixed.expand(linear_solver.solve(-fixed.reduce(B[:, j].copy())), sensitivity)
            force_derivative = B[:, j] + K @ sensitivity
            if np.linalg.norm(fixed.reduce(force_derivative), ord=np.inf) > 1e-8:
                raise RuntimeError("Sensitivity equation residual too large")
            derivative[j] = force_derivative[loaded].sum()
        if abs(derivative @ theta - force) > 1e-7 * max(1, abs(force)):
            raise RuntimeError("Reaction sensitivity violates stiffness scaling")
        return force, derivative

    def measurements(theta, need_jacobian):
        results = [reaction(theta, 0, 0.08, need_jacobian), reaction(theta, 2, -0.04, need_jacobian)]
        if any(value is None for value in results):
            return None
        return np.array([value[0] for value in results]), np.array([value[1] for value in results])

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
    result = solver.gauss_newton_solve(residual, np.zeros(2), solver.GaussNewtonOptions(gradient_tolerance=1e-8))
    fitted = np.exp(result.x)
    r = residual.eval_value(result.x)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    reference = reaction(truth, 2, -0.07, False)
    prediction = reaction(fitted, 2, -0.07, False)
    if reference is None or prediction is None or r is None:
        raise RuntimeError("Validation solve failed")
    prediction_error = abs(prediction[0] - reference[0]) / abs(reference[0])
    print(f"truth={truth[0]:.6g} {truth[1]:.6g}, fitted={fitted[0]:.6g} {fitted[1]:.6g}, "
          f"parameter error={parameter_error:.6g}, residual={np.linalg.norm(r):.6g}, "
          f"held-out error={prediction_error:.6g}, forward solves={solves}")
    return 0 if (result.status == solver.GaussNewtonStatus.Converged and parameter_error < 1e-4
                 and np.linalg.norm(r) < 1e-7 and prediction_error < 1e-4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
