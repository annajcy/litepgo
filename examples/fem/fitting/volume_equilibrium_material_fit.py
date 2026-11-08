"""Fit volume material parameters from reactions using analytic equilibrium sensitivities.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/fitting/volume_equilibrium_material_fit.py
"""

from litepgo.energy import AffineResponse
from pathlib import Path
from dataclasses import dataclass
from functools import partial

import numpy as np

from litepgo import energy, dof, fem, mesh, solver


ElasticEnergy = fem.VolumetricFemEnergy[
    fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
]
ForceParameterResponse = fem.VolumetricFemForceMaterialResponse[
    fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
]


@dataclass
class Problem:
    mapping: dof.VertexDofMap
    geometry: list
    plastic: list
    parameter_map: dof.ConstantDofMap
    fixed: dof.DofReduction
    top: np.ndarray
    height: float
    solves: int = 0

    def make_energy(self, theta):
        return ElasticEnergy(
            self.mapping,
            self.geometry,
            self.plastic,
            [AffineResponse(np.zeros((2, 3)), theta)] * len(self.geometry),
        )

    def make_response(self, u):
        return ForceParameterResponse(
            self.parameter_map, self.mapping, self.geometry, u, self.plastic
        )


def make_geometry(input_path=None):
    rest = mesh.read_veg(
        input_path or Path(__file__).resolve().parents[3] / "assets/box-tet.veg"
    )
    topology = rest.topology
    positions = rest.positions.ravel()
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedResponse[fem.TetP1Basis]
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
    return Problem(
        mapping,
        geometry,
        [AffineResponse(np.zeros((9, 3)), np.eye(3).ravel())] * len(geometry),
        dof.ConstantDofMap(topology, 2),
        fixed,
        top,
        height,
    )


def reaction(problem, theta, axis, strain, need_jacobian):
    if not np.isfinite(theta).all() or (theta <= 0).any():
        return None
    initial = np.zeros(problem.mapping.num_dofs)
    loaded = 3 * problem.top + axis
    initial[loaded] = strain * problem.height
    equilibrium = solver.StaticEquilibriumResponse(
        2,
        problem.fixed,
        initial,
        solver.NewtonOptions(gradient_tolerance=1e-10, max_iterations=100),
        problem.make_energy,
        problem.make_response,
    )
    displacement, J = (equilibrium.eval_value_and_jacobian(theta) if need_jacobian else (equilibrium.eval_value(theta), None))
    problem.solves += 1
    elastic = problem.make_energy(theta)
    force = elastic.eval_gradient(displacement)[loaded].sum()
    if not need_jacobian:
        return force, None
    # Total reaction derivative: direct material dependence plus equilibrium motion.
    force_derivative = (
        problem.make_response(displacement).eval_jacobian(theta)
        + elastic.eval_hessian(displacement) @ J
    )
    return force, np.asarray(force_derivative[loaded].sum(axis=0)).ravel()


def measurements(measure, theta, need_jacobian):
    results = [
        measure(theta, 0, 0.08, need_jacobian),
        measure(theta, 2, -0.04, need_jacobian),
    ]
    if any(value is None for value in results):
        return None
    return np.array([value[0] for value in results]), np.array(
        [value[1] for value in results]
    )


class MaterialResidual(energy.FixedPatternResponse):
    def __init__(self, measurement, observed, scales):
        super().__init__(np.ones((2, 2)))
        self.measurement, self.observed, self.scales = measurement, observed, scales

    def eval_impl(self, theta, need_jacobian):
        values = self.measurement(theta, need_jacobian)
        if values is None:
            return None
        forces, J = values
        residual = (forces - self.observed) / self.scales
        return (residual, J / self.scales[:, None]) if need_jacobian else residual


def make_problem(input_path=None):
    problem = make_geometry(input_path)
    measure = partial(reaction, problem)
    measurement = partial(measurements, measure)
    truth = np.array([2.0, 3.0])
    observed = measurement(truth, False)[0]
    scales = np.maximum(np.abs(observed), 1e-8)
    residual = energy.ComposedResponse(
        MaterialResidual(measurement, observed, scales), energy.ExpResponse(2)
    )
    return residual, truth, measure, lambda: problem.solves


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
    return 0 if result.status == solver.GaussNewtonStatus.Converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
