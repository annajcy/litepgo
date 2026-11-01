"""Fit two shared positive in-plane Fp stretches at a prescribed shell pose.

Python mirror of shell_fixed_pose_fp_fit.cpp. Geometry, displacement and material
parameters stay fixed; no equilibrium solve or plastic evolution is involved.
"""

from pathlib import Path

import numpy as np

from litepgo import dof, energy, fem, mesh, solver


def main():
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
    rest = [fem.TriangleMidEdgeRestData(mapping.gather(e, nodes.ravel()), stencils.opposite[e])
            for e in range(mapping.num_elements)]
    thickness = [0.1] * mapping.num_elements

    # Anisotropic stretch, shear and saddle bending excite both in-plane plastic stretches.
    X = nodes - center
    u = np.column_stack([.08 * X[:, 0] + .04 * X[:, 1], -.03 * X[:, 1],
                         .15 * X[:, 0] * X[:, 1] / size]).ravel()
    truth, initial = np.array([1.06, .95]), np.ones(2)
    theta = [np.array([2., 3.])] * mapping.num_elements
    true_Fp = np.diag([*truth, 1.])
    # Independent energy gradients: retain x=X+u and ORIGINAL integration areas.
    target = np.zeros(mapping.num_dofs)
    for e, reference in enumerate(rest):
        Y = reference.X_local @ true_Fp.T
        natural = fem.TriangleMidEdgeRestData(Y, reference.opposite)
        natural_u = reference.X_local + mapping.gather(e, u) - Y
        local = fem.ShellMaterialLocalEnergy2D[fem.KoiterStVK](
            natural, natural_u, thickness[e], theta[e])
        target[np.array(mapping.dof_indices(e))] += (
            .5 * reference.sqrt_det_a_bar * local.eval_gradient(np.zeros(2)))
    scale = np.linalg.norm(target)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Calibration requires finite nonzero target forces")
    force = fem.ShellFemForceFpResponse[fem.KoiterStVK](
        dof.ConstantDofMap(topology, 9), mapping, rest, u, thickness, theta)

    class PlasticResidual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((force.num_outputs, 2)))

        def eval_values(self, p, need_jacobian):
            with np.errstate(over="ignore", under="ignore"):
                stretches = np.exp(p)
            if not np.isfinite(stretches).all() or (stretches <= 0).any():
                return None  # Reject unrepresentable trials.
            packed = np.diag([*stretches, 1.]).ravel(order="F")
            value, full_J = force.eval(packed, need_jacobian=need_jacobian)
            residual = (value - target) / scale
            if not need_jacobian:
                return residual
            # Column-major diagonal entries 0, 4; d(exp(p_j))/dp_j = exp(p_j).
            J = full_J[:, [0, 4]].multiply(stretches[None, :] / scale).tocsc()
            return residual, J

    residual = PlasticResidual()

    def check_identifiability(p):
        singular = np.linalg.svd(residual.eval_jacobian(p).toarray(), compute_uv=False)
        ratio = singular[1] / singular[0]
        if not ratio > 1e-6:
            raise RuntimeError("Prescribed pose cannot locally identify both in-plane plastic stretches")
        return ratio

    initial_p = np.log(initial)
    initial_ratio = check_identifiability(initial_p)
    truth_ratio = check_identifiability(np.log(truth))
    # Check the composed Jacobian, including the exponential parameterization.
    check_p = np.log(truth)
    J = residual.eval_jacobian(check_p).toarray()
    fd = np.empty_like(J)
    for j in range(2):
        plus, minus = check_p.copy(), check_p.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        fd[:, j] = (residual.eval_value(plus) - residual.eval_value(minus)) / 2e-6
    jacobian_error = np.linalg.norm(J - fd) / np.linalg.norm(J)
    if not jacobian_error < 1e-6:
        raise RuntimeError("Plastic stretch chain-rule check failed")
    result = solver.gauss_newton_solve(residual, initial_p,
                                     solver.GaussNewtonOptions(gradient_tolerance=1e-12))
    fitted = np.exp(result.x)
    fitted_ratio = check_identifiability(result.x)
    parameter_error = np.linalg.norm(fitted - truth) / np.linalg.norm(truth)
    relative_residual = np.linalg.norm(residual.eval_value(result.x))
    converged = result.status == solver.GaussNewtonStatus.Converged
    print(f"truth={truth[0]:.12g} {truth[1]:.12g}, "
          f"initial={initial[0]:.12g} {initial[1]:.12g}, "
          f"fitted={fitted[0]:.12g} {fitted[1]:.12g}, converged={int(converged)}, "
          f"iterations={result.iterations}, parameter_error={parameter_error:.12g}, "
          f"force_residual={scale*relative_residual:.12g}, relative_residual={relative_residual:.12g}, "
          f"initial_ratio={initial_ratio:.12g}, truth_ratio={truth_ratio:.12g}, "
          f"fitted_ratio={fitted_ratio:.12g}, jacobian_error={jacobian_error:.12g}")
    return 0 if converged and parameter_error < 1e-7 and relative_residual < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
