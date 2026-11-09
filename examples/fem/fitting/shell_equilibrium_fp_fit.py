"""Fit two shared in-plane log plastic stretches from equilibrium displacement and reaction.

Geometry, material theta=(2,3), and boundary displacements are fixed.
Run: uv run --no-sync python examples/fem/fitting/shell_equilibrium_fp_fit.py
"""

from dataclasses import dataclass
from functools import partial

import numpy as np
from litepgo import dof, energy, fem, mesh, solver

ElasticEnergy = fem.ShellFemEnergy
ForceFpResponse = fem.ShellFemForceFpResponse


@dataclass
class Problem:
    displacement_map: dof.StencilDofMap
    rest: list
    thickness: list
    theta: list
    fixed: dof.DofReduction
    prescribed: np.ndarray
    parameter_map: dof.ConstantDofMap
    physical_map: dof.ElementDofMap


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
    theta = [np.array([2.0, 3.0])] * mapping.num_elements
    return Problem(
        mapping,
        rest,
        thickness,
        theta,
        fixed,
        prescribed,
        dof.ConstantDofMap(topology, 2),
        dof.ElementDofMap(topology, 9),
    )


def make_factories(problem):
    # Both factories use the same p -> per-element Fp transform.
    plastic = energy.ElementwiseResponse(
        problem.parameter_map, fem.LogDiagonalFpResponse(2)
    )

    def make_energy(p):
        packed = plastic.eval_value(p).reshape(-1, 9)
        fields = [row.reshape(3, 3, order="F") for row in packed]
        return ElasticEnergy(fem.PointwiseShellAdapter[fem.KoiterStVK](),
            problem.displacement_map,
            problem.rest,
            problem.thickness,
            problem.theta,
            fields,
        )

    def make_response(u):
        force = ForceFpResponse(fem.PointwiseShellAdapter[fem.KoiterStVK](),
            problem.physical_map,
            problem.displacement_map,
            problem.rest,
            u,
            problem.thickness,
            problem.theta,
        )
        return energy.ComposedResponse(force, plastic)

    return make_energy, make_response


def measure(problem, make_energy, make_response, p, load=1.0, need_jacobian=False):
    # 1. Solve equilibrium, starting from the prescribed displacement every time.
    equilibrium = solver.StaticEquilibriumResponse(
        2,
        problem.fixed,
        load * problem.prescribed,
        solver.NewtonOptions(gradient_tolerance=1e-13, max_iterations=100),
        make_energy,
        make_response,
    )
    displacement, state_jacobian = (equilibrium.eval_value_and_jacobian(p) if need_jacobian else (equilibrium.eval_value(p), None))

    # 2. Compute force and its total derivative along the equilibrium branch.
    elastic = make_energy(p)
    force = elastic.eval_gradient(displacement)
    reaction_jacobian = None
    if need_jacobian:
        force_parameter_jacobian = make_response(displacement).eval_jacobian(p)
        stiffness = elastic.eval_hessian(displacement)
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
        super().__init__(np.ones((len(target), 2)))
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
    truth = np.array([1.06, 0.95])
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
        residual, np.zeros(2), solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    )
    converged = result.status == solver.GaussNewtonStatus.Converged
    fitted = np.exp(result.x)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    relative_residual = np.linalg.norm(residual.eval_value(result.x))

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
