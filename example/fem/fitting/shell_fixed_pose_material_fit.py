"""Recover shared KoiterStVK parameters from forces at a prescribed shell pose.

Python mirror of shell_fixed_pose_material_fit.cpp. Geometry, displacement and thickness
stay fixed; no equilibrium solve, noise or regularization is involved.
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

    # Anisotropic stretch, shear and saddle bending excite both material parameters.
    X = nodes - center
    u = np.column_stack([.08 * X[:, 0] + .04 * X[:, 1], -.03 * X[:, 1],
                         .15 * X[:, 0] * X[:, 1] / size]).ravel()
    truth, initial = np.array([2., 3.]), np.array([.7, 1.2])
    elastic = fem.ShellFemEnergy[fem.KoiterStVK](mapping, rest, thickness, [truth] * mapping.num_elements)
    # Generate observations through the energy API, independently of the response path.
    target = elastic.eval_gradient(u)
    scale = np.linalg.norm(target)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Calibration requires finite nonzero target forces")
    force = fem.ShellFemForceMaterialResponse[fem.KoiterStVK](
        dof.ConstantDofMap(topology, 2), mapping, rest, u, thickness)
    singular = np.linalg.svd(force.eval_jacobian(initial).toarray(), compute_uv=False)
    singular_ratio = singular[1] / singular[0]
    if not singular_ratio > 1e-6:
        raise RuntimeError("Prescribed pose cannot identify both material parameters")

    class MaterialResidual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(force.jacobian_pattern)

        def eval_values(self, theta, need_jacobian):
            # None rejects infeasible trials without changing parameterization.
            if not np.isfinite(theta).all() or theta[1] <= 0 or theta.sum() <= 0:
                return None
            value, J = force.eval(theta, need_jacobian=need_jacobian)
            residual = (value - target) / scale
            return (residual, J / scale) if need_jacobian else residual

    result = solver.gauss_newton_solve(MaterialResidual(), initial,
                                     solver.GaussNewtonOptions(gradient_tolerance=1e-12))
    parameter_error = np.linalg.norm(result.x - truth) / np.linalg.norm(truth)
    force_residual = np.linalg.norm(force.eval_value(result.x) - target)
    converged = result.status == solver.GaussNewtonStatus.Converged
    print(f"truth={truth[0]:.12g} {truth[1]:.12g}, initial={initial[0]:.12g} {initial[1]:.12g}, "
          f"fitted={result.x[0]:.12g} {result.x[1]:.12g}, converged={int(converged)}, "
          f"iterations={result.iterations}, parameter_error={parameter_error:.12g}, "
          f"force_residual={force_residual:.12g}, relative_residual={force_residual/scale:.12g}, "
          f"singular_ratio={singular_ratio:.12g}")
    return 0 if converged and parameter_error < 1e-7 and force_residual/scale < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
