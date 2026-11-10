"""Fit three shared log plastic stretches from equilibrium displacement and reaction.

Geometry, material theta=(2,3), and boundary displacements are fixed.
Run: uv run --no-sync python examples/fem/fitting/volume_equilibrium_fp_fit.py
"""

from litepgo.energy import AffineResponse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from litepgo import dof, energy, fem, mesh, solver

ElasticEnergy = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]
ForceFpResponse = fem.VolumetricFemForceFpResponse[fem.TetP1Basis, fem.TetCentroidQuadrature]


@dataclass
class Problem:
    displacement_map: dof.VertexDofMap
    geometry: list
    theta: list
    fixed: dof.DofReduction
    prescribed: np.ndarray
    parameter_map: dof.ConstantDofMap
    physical_map: dof.ElementDofMap


def make_problem(input_path=None):
    input_path = (
        input_path or Path(__file__).resolve().parents[3] / "assets/box-tet.veg"
    )
    volume = mesh.read_veg(input_path)
    topology, nodes = volume.topology, volume.positions
    displacement_map = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedResponse[fem.TetP1Basis]
    geometry = [
        XField(displacement_map.gather(e, nodes.ravel()))
        for e in range(displacement_map.num_elements)
    ]
    theta = [AffineResponse(np.zeros((2, 3)), np.array([2.0, 3.0])) for _ in geometry]

    # Clamp the bottom; prescribe shear and extension on the top.
    z = nodes[:, 2]
    height = z.max() - z.min()
    if not height > 0:
        raise RuntimeError("Require nonempty volume with positive height")
    top = np.flatnonzero(np.abs(z - z.max()) < height * 1e-8)
    boundary = np.flatnonzero(
        (np.abs(z - z.min()) < height * 1e-8) | (np.abs(z - z.max()) < height * 1e-8)
    )
    fixed_ids = (3 * boundary[:, None] + np.arange(3)).ravel()
    fixed = dof.DofReduction(nodes.size, fixed_ids.tolist())
    if not len(fixed.free_dofs) or not len(fixed_ids):
        raise RuntimeError("Require both fixed and free DOFs")
    prescribed = np.zeros(nodes.size)
    prescribed[3 * top] = 0.04 * height
    prescribed[3 * top + 2] = 0.05 * height
    return Problem(
        displacement_map,
        geometry,
        theta,
        fixed,
        prescribed,
        dof.ConstantDofMap(topology, 3),
        dof.ElementDofMap(topology, 9),
    )


def make_factories(problem):
    # Both factories use the same p -> per-element Fp transform.
    plastic = energy.ElementwiseResponse(
        problem.parameter_map, fem.LogDiagonalFpResponse(3)
    )

    def make_energy(p):
        packed = plastic.evaluate(p).value.reshape(-1, 9)
        fields = [AffineResponse(np.zeros((9, 3)), row) for row in packed]
        return ElasticEnergy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
            problem.displacement_map, problem.geometry, fields, problem.theta
        )

    def make_response(u):
        force = ForceFpResponse(fem.PointwiseVolumetricAdapter[fem.NeoHookean](),
            problem.physical_map,
            problem.displacement_map,
            problem.geometry,
            u,
            problem.theta,
        )
        return energy.ComposedResponse(force, plastic)

    return make_energy, make_response


def measure(problem, make_energy, make_response, p, load=1.0, need_jacobian=False):
    # 1. Solve equilibrium, starting from the prescribed displacement every time.
    equilibrium = solver.StaticEquilibriumResponse(
        3,
        problem.fixed,
        load * problem.prescribed,
        solver.NewtonOptions(gradient_tolerance=1e-11, max_iterations=100),
        make_energy,
        make_response,
    )
    evaluation = equilibrium.evaluate(p)
    state_jacobian = evaluation.jacobian() if need_jacobian else None
    displacement = evaluation.value

    # 2. Compute force and its total derivative along the equilibrium branch.
    elastic = make_energy(p)
    force = elastic.evaluate(displacement).gradient()
    reaction_jacobian = None
    if need_jacobian:
        force_parameter_jacobian = make_response(displacement).evaluate(p).jacobian()
        stiffness = elastic.evaluate(displacement).hessian()
        reaction_jacobian = force_parameter_jacobian + stiffness @ state_jacobian

    # 3. Observe free displacement, followed by the reactions at fixed DOFs.
    free_ids, fixed_ids = problem.fixed.free_dofs, problem.fixed.fixed_dofs
    value = np.concatenate([displacement[free_ids], force[fixed_ids]])
    jacobian = None
    if need_jacobian:
        jacobian = np.vstack(
            [state_jacobian.toarray()[free_ids], reaction_jacobian.toarray()[fixed_ids]]
        )
    return value, jacobian


class Residual(energy.FixedPatternResponse):
    def __init__(self, measurement, target, scales):
        super().__init__(np.ones((len(target), 3)))
        self.measurement, self.target, self.scales = measurement, target, scales

    def eval_impl(self, p, need_jacobian):
        prediction, jacobian = self.measurement(p, need_jacobian=need_jacobian)
        residual = (prediction - self.target) / self.scales
        return (
            (residual, jacobian / self.scales[:, None]) if need_jacobian else residual
        )


def main():
    problem = make_problem()
    make_energy, make_response = make_factories(problem)
    measurement = partial(measure, problem, make_energy, make_response)

    # Synthetic target; each observation block uses a fixed normalization.
    truth = np.array([1.04, 0.97, 1.02])
    target, _ = measurement(np.log(truth))
    num_free = len(problem.fixed.free_dofs)
    scales = np.concatenate(
        [
            np.full(num_free, max(np.linalg.norm(target[:num_free]), 1e-8)),
            np.full(
                len(target) - num_free, max(np.linalg.norm(target[num_free:]), 1e-8)
            ),
        ]
    )
    residual = Residual(measurement, target, scales)

    result = solver.gauss_newton_solve(
        residual, np.zeros(3), solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    )
    converged = result.status == solver.GaussNewtonStatus.Converged
    fitted = np.exp(result.x)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    relative_residual = np.linalg.norm(residual.evaluate(result.x).value)

    # Predict a load that was not used for fitting.
    held_out, _ = measurement(np.log(truth), load=0.6)
    prediction, _ = measurement(result.x, load=0.6)
    held_out_error = np.linalg.norm((prediction - held_out) / scales)
    print(
        "truth="
        + " ".join(f"{x:.12g}" for x in truth)
        + ", fitted="
        + " ".join(f"{x:.12g}" for x in fitted)
        + f", converged={int(converged)}, parameter_error={parameter_error:.12g}"
        + f", relative_residual={relative_residual:.12g}, held_out_error={held_out_error:.12g}"
    )
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
