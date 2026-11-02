"""Fit plane-stress KoiterStVK parameters from equilibrium boundary reactions.

A small triangulated patch has prescribed boundary displacements and a free
interior vertex. Thickness is fixed; observations are synthetic and noiseless.

Setup: uv pip install --python .venv/bin/python torch
Run: .venv/bin/python examples/fem/fitting/shell_equilibrium_material_fit_torch.py
Requires a built litepgo Python package. Uses CPU float64 and Torch L-BFGS.
"""

import numpy as np
import torch
from litepgo import torch as lp_torch

from litepgo import dof, energy, fem, mesh, solver


def main():
    # Build the mesh, FEM model and prescribed boundary displacement.
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
    ElasticEnergy = fem.ShellFemEnergy[fem.KoiterStVK]
    ForceParameterResponse = fem.ShellFemForceMaterialResponse[fem.KoiterStVK]

    # Solve equilibrium before measuring reactions; free DOFs move with the parameters.
    def experiment(theta, load, need_jacobian):
        initial = load * prescribed
        equilibrium = solver.StaticEquilibriumResponse(
            2,
            fixed,
            initial,
            solver.NewtonOptions(gradient_tolerance=1e-12, max_iterations=100),
            lambda parameters: ElasticEnergy(
                mapping, rest, thickness, [parameters] * mapping.num_elements
            ),
            lambda parameters, displacement: ForceParameterResponse(
                dof.ConstantDofMap(topology, 2), mapping, rest, displacement, thickness
            ).eval_jacobian(parameters),
        )
        displacement, jacobian = equilibrium.eval(theta, need_jacobian=need_jacobian)
        elastic = ElasticEnergy(
            mapping, rest, thickness, [theta] * mapping.num_elements
        )
        gradient = elastic.eval_gradient(displacement)
        if (
            not np.array_equal(displacement[fixed_ids], initial[fixed_ids])
            or np.max(np.abs(gradient[12:15])) > 2e-12
        ):
            raise RuntimeError("Invalid equilibrium or changed boundary values")
        reaction_J = None
        if need_jacobian:
            B = ForceParameterResponse(
                dof.ConstantDofMap(topology, 2), mapping, rest, displacement, thickness
            ).eval_jacobian(theta)
            # Total reaction derivative includes the equilibrium displacement sensitivity.
            reaction_J = (B + elastic.eval_hessian(displacement) @ jacobian).toarray()[
                fixed_ids
            ]
        return gradient[fixed_ids], reaction_J, displacement, jacobian

    truth, initial = np.array([2.0, 3.0]), np.array([0.7, 1.2])
    target = experiment(truth, 1.0, False)[0]
    scale = np.linalg.norm(target)
    _, tangent, _, state_J = experiment(initial, 1.0, True)
    singular = np.linalg.svd(tangent, compute_uv=False)
    singular_ratio = singular[1] / singular[0]
    if (
        not scale > 0
        or not singular_ratio > 1e-6
        or not np.linalg.norm(state_J.toarray()) > 1e-5
    ):
        raise RuntimeError(
            "Experiment must excite both parameters and equilibrium sensitivity"
        )
    fd_error = 0.0
    for h in (1e-3, 1e-4):
        for c in range(2):
            delta = np.eye(2)[c] * h
            rp, _, up, _ = experiment(initial + delta, 1.0, False)
            rm, _, um, _ = experiment(initial - delta, 1.0, False)
            fd_error = max(
                fd_error,
                np.max(np.abs((rp - rm) / (2 * h) - tangent[:, c])),
                np.max(np.abs((up - um) / (2 * h) - state_J.toarray()[:, c])),
            )
    if fd_error > 2e-7:
        raise RuntimeError(
            "Equilibrium/reaction derivatives failed re-solve finite differences"
        )

    # Normalize observations so the loss compares relative measurement errors.
    class Residual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((len(fixed_ids), 2)))

        def eval_values(self, parameters, need_jacobian):
            if parameters[1] <= 0 or parameters.sum() <= 0:
                return None
            reaction, jacobian, _, _ = experiment(parameters, 1.0, need_jacobian)
            r = (reaction - target) / scale
            return (r, jacobian / scale) if need_jacobian else r

    residual = Residual()
    optimization_parameters = torch.tensor(
        initial, dtype=torch.float64, requires_grad=True
    )
    # L-BFGS may evaluate this closure repeatedly during its line search.
    optimizer = torch.optim.LBFGS(
        [optimization_parameters],
        max_iter=150,
        tolerance_grad=1e-10,
        tolerance_change=1e-24,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = 0.5 * lp_torch.response(residual, optimization_parameters).square().sum()
        # The adapter applies the FEM Jacobian transpose to the upstream gradient.
        loss.backward()
        return loss

    optimizer.step(closure)
    closure()
    converged = optimization_parameters.grad.abs().max().item() <= 1e-10
    fitted_parameters = optimization_parameters.detach().numpy().copy()
    parameter_error = np.linalg.norm(fitted_parameters - truth) / np.linalg.norm(truth)
    relative_residual = (
        np.linalg.norm(experiment(fitted_parameters, 1.0, False)[0] - target) / scale
    )
    # Validate on a load that was not used to construct the fitting target.
    held_out = experiment(truth, 0.6, False)[0]
    held_out_error = np.linalg.norm(
        experiment(fitted_parameters, 0.6, False)[0] - held_out
    ) / np.linalg.norm(held_out)
    print(
        f"fitted={fitted_parameters[0]:.12g} {fitted_parameters[1]:.12g}, converged={int(converged)}, "
        f"parameter_error={parameter_error:.12g}, relative_residual={relative_residual:.12g}, "
        f"held_out_error={held_out_error:.12g}, fd_error={fd_error:.12g}, singular_ratio={singular_ratio:.12g}"
    )
    return (
        0
        if converged
        and parameter_error < 1e-6
        and relative_residual < 1e-8
        and held_out_error < 1e-8
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
