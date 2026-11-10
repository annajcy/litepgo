"""Fit two material parameters from equilibrium reactions.

Geometry, thickness, and boundary displacements are fixed.
Run: uv run --no-sync python examples/fem/fitting/shell_equilibrium_material_fit.py
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
from litepgo import dof, energy, fem, mesh, solver



@dataclass
class Problem:
    displacement_map: dof.StencilDofMap
    rest: list
    thickness: list
    fixed: dof.DofReduction
    prescribed: np.ndarray
    parameter_map: dof.ConstantDofMap


def make_problem():
    nodes = np.array([[x, y, 0.0] for y in (-1.0, 0.0, 1.0) for x in (-1.0, 0.0, 1.0)])
    faces = []
    for y in range(2):
        for x in range(2):
            a = 3 * y + x
            faces.extend([[a, a + 1, a + 4], [a, a + 4, a + 3]])
    topology = mesh.TriangleTopology(9, np.array(faces, dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(9, stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(
            mapping.gather(e, nodes.ravel()), stencils.opposite[e]
        )
        for e in range(mapping.num_elements)
    ]
    thickness = [0.1] * mapping.num_elements
    fixed_ids = np.array([i for i in range(27) if i // 3 != 4])
    fixed = dof.DofReduction(27, fixed_ids.tolist())
    X, Y = nodes[:, 0], nodes[:, 1]
    prescribed = np.column_stack(
        [
            0.08 * X + 0.025 * X**2 + 0.04 * Y,
            0.03 * Y + 0.02 * X * Y,
            0.08 * X * Y + 0.03 * X**2,
        ]
    ).ravel()
    return Problem(
        mapping, rest, thickness, fixed, prescribed, dof.ConstantDofMap(topology, 2)
    )


def make_factories(problem, material=None):
    """Energy and force-parameter response factories sharing one batch material."""
    if material is None:
        material = fem.PointwiseShellAdapter[fem.KoiterStVK]()
    ElasticEnergy = fem.ShellFemEnergy
    ForceParameterResponse = fem.ShellFemForceMaterialResponse

    def make_energy(theta):
        return ElasticEnergy(material,
            problem.displacement_map,
            problem.rest,
            problem.thickness,
            [theta] * problem.displacement_map.num_elements,
        )

    def make_response(u):
        return ForceParameterResponse(material,
            problem.parameter_map,
            problem.displacement_map,
            problem.rest,
            u,
            problem.thickness,
        )

    return make_energy, make_response


def measure(problem, make_energy, make_response, p, load=1.0, need_jacobian=False):
    # 1. Solve equilibrium, starting from the prescribed displacement every time.
    equilibrium = solver.StaticEquilibriumResponse(
        2,
        problem.fixed,
        load * problem.prescribed,
        solver.NewtonOptions(gradient_tolerance=1e-12, max_iterations=100),
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

    # Observe constrained reactions; retain state derivatives for analysis/tests.
    fixed_ids = problem.fixed.fixed_dofs
    return Observation(
        force[fixed_ids],
        reaction_jacobian.toarray()[fixed_ids] if need_jacobian else None,
        displacement,
        state_jacobian,
    )


@dataclass
class Observation:
    reaction: np.ndarray
    jacobian: object
    displacement: np.ndarray
    state_jacobian: object


class Residual(energy.FixedPatternResponse):
    def __init__(self, measurement, target, scales):
        super().__init__(np.ones((len(target), 2)))
        self.measurement, self.target, self.scales = measurement, target, scales

    def eval_impl(self, theta, need_jacobian):
        if theta[1] <= 0 or theta.sum() <= 0:
            return None
        observation = self.measurement(theta, need_jacobian=need_jacobian)
        residual = (observation.reaction - self.target) / self.scales
        return (
            (residual, observation.jacobian / self.scales)
            if need_jacobian
            else residual
        )


def main():
    problem = make_problem()
    make_energy, make_response = make_factories(problem)
    measurement = partial(measure, problem, make_energy, make_response)

    truth, initial = np.array([2.0, 3.0]), np.array([0.7, 1.2])
    target = measurement(truth).reaction
    scale = np.linalg.norm(target)
    if not scale > 0:
        raise RuntimeError("Calibration requires nonzero target reactions")
    residual = Residual(measurement, target, scale)
    result = solver.gauss_newton_solve(
        residual, initial, solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    )
    converged = result.status == solver.GaussNewtonStatus.Converged
    parameter_error = np.linalg.norm(result.x - truth) / np.linalg.norm(truth)
    relative_residual = np.linalg.norm(residual.evaluate(result.x).value)
    held_out = measurement(truth, load=0.6).reaction
    prediction = measurement(result.x, load=0.6).reaction
    held_out_error = np.linalg.norm(prediction - held_out) / np.linalg.norm(held_out)
    print(
        f"fitted={result.x[0]:.12g} {result.x[1]:.12g}, converged={int(converged)}, "
        f"parameter_error={parameter_error:.12g}, relative_residual={relative_residual:.12g}, "
        f"held_out_error={held_out_error:.12g}"
    )
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
