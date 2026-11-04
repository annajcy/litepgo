"""Fit two shared positive in-plane Fp stretches at a prescribed shell pose.

Python mirror of shell_fixed_pose_fp_fit.cpp. Geometry, displacement and material
parameters stay fixed; no equilibrium solve or plastic_matrix evolution is involved.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/fitting/shell_fixed_pose_fp_fit.py
"""

from pathlib import Path

import numpy as np

from litepgo import dof, energy, fem, mesh, solver


class Residual(energy.FixedPatternResponse):
    def __init__(self, force, target, scale):
        super().__init__(force.jacobian_pattern)
        self.force, self.target, self.scale = force, target, scale

    def eval_values(self, p, need_jacobian):
        value, J = self.force.eval(p, need_jacobian=need_jacobian)
        residual = (value - self.target) / self.scale
        return (residual, J / self.scale) if need_jacobian else residual


def make_problem():
    sheet = mesh.read_obj(Path(__file__).resolve().parents[3] / "assets/shell.obj")
    topology, nodes = sheet.topology, sheet.positions
    if topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty sheet")
    lower, upper = nodes.min(axis=0), nodes.max(axis=0)
    center = (lower + upper) / 2
    size = (upper - lower)[:2].max()
    tolerance = size * 1e-8
    if not ((upper - lower)[:2] > tolerance).all() or upper[2] - lower[2] > tolerance:
        raise RuntimeError("Example requires a flat sheet in the xy plane")
    stencils = mesh.make_triangle_mid_edge_stencils(topology)
    mapping = dof.StencilDofMap(topology.num_vertices, stencils.vertices, 3)
    rest = [
        fem.TriangleMidEdgeRestData(
            mapping.gather(e, nodes.ravel()), stencils.opposite[e]
        )
        for e in range(mapping.num_elements)
    ]
    thickness = [0.1] * mapping.num_elements

    # Anisotropic stretch, shear and saddle bending excite both in-plane plastic_matrix stretches.
    X = nodes - center
    displacement = np.column_stack(
        [
            0.08 * X[:, 0] + 0.04 * X[:, 1],
            -0.03 * X[:, 1],
            0.15 * X[:, 0] * X[:, 1] / size,
        ]
    ).ravel()
    truth, initial = np.array([1.06, 0.95]), np.ones(2)
    theta = [np.array([2.0, 3.0])] * mapping.num_elements
    true_Fp = np.diag([*truth, 1.0])
    # Independent energy gradients: retain x=X+displacement and ORIGINAL integration areas.
    target = np.zeros(mapping.num_dofs)
    for e, reference in enumerate(rest):
        Y = reference.X_local @ true_Fp.T
        natural = fem.TriangleMidEdgeRestData(Y, reference.opposite)
        natural_u = reference.X_local + mapping.gather(e, displacement) - Y
        local = fem.ShellMaterialLocalEnergy2D[fem.KoiterStVK](
            natural, natural_u, thickness[e], theta[e]
        )
        target[np.array(mapping.dof_indices(e))] += (
            0.5 * reference.sqrt_det_a_bar * local.eval_gradient(np.zeros(2))
        )
    scale = np.linalg.norm(target)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Calibration requires finite nonzero target forces")
    force = fem.ShellFemForceFpResponse[fem.KoiterStVK](
        dof.ConstantDofMap(topology, 9), mapping, rest, displacement, thickness, theta
    )

    parameterized_force = energy.ComposedResponse(force, fem.LogDiagonalFpResponse(2))
    return Residual(parameterized_force, target, scale), truth, initial, scale


def main():
    residual, truth, initial, scale = make_problem()
    initial_p = np.log(initial)
    # Fit the observation residual using its analytic Jacobian.
    result = solver.gauss_newton_solve(
        residual, initial_p, solver.GaussNewtonOptions(gradient_tolerance=1e-12)
    )
    fitted = np.exp(result.x)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    relative_residual = np.linalg.norm(residual.eval_value(result.x))
    converged = result.status == solver.GaussNewtonStatus.Converged
    print(
        f"truth={truth[0]:.12g} {truth[1]:.12g}, "
        f"initial={initial[0]:.12g} {initial[1]:.12g}, "
        f"fitted={fitted[0]:.12g} {fitted[1]:.12g}, converged={int(converged)}, "
        f"iterations={result.iterations}, parameter_error={parameter_error:.12g}, "
        f"force_residual={scale * relative_residual:.12g}, relative_residual={relative_residual:.12g}"
    )
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
