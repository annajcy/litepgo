"""Fit three shared positive diagonal Fp stretches at a prescribed tetrahedral pose.

Python mirror of volume_fixed_pose_fp_fit.cpp. Geometry, displacement and material
parameters stay fixed; no equilibrium solve or plastic evolution is involved.
"""

from pathlib import Path

import numpy as np

from litepgo import dof, energy, fem, mesh, solver


def main():
    volume = mesh.read_veg(Path(__file__).resolve().parents[3] / "assets/box-tet.veg")
    topology, nodes = volume.topology, volume.positions
    if topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty tetrahedral mesh")
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedField3D[fem.TetP1Basis]
    geometry = [XField(mapping.gather(e, nodes.ravel())) for e in range(mapping.num_elements)]
    F = np.array([[1.12, .08, .02], [.01, .96, .06], [.03, 0., 1.07]])
    u = ((nodes - nodes.mean(axis=0)) @ (F - np.eye(3)).T).ravel()
    truth, initial = np.array([1.08, .94, 1.03]), np.ones(3)
    theta = [fem.IdentityField3D(np.array([2., 3.])) for _ in geometry]
    Elastic = fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
    elastic = Elastic(mapping, geometry,
                      [fem.IdentityField3D(np.diag(truth).ravel(order="F")) for _ in geometry], theta)
    # Target comes from the energy gradient, independently of the force-response path.
    target = elastic.eval_gradient(u)
    scale = np.linalg.norm(target)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Calibration requires finite nonzero target forces")
    Force = fem.VolumetricFemForceFpResponse[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
    force = Force(dof.ConstantDofMap(topology, 9), mapping, geometry, u, theta)

    class PlasticResidual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((force.num_outputs, 3)))

        def eval_values(self, p, need_jacobian):
            with np.errstate(over="ignore", under="ignore"):
                stretches = np.exp(p)
            if not np.isfinite(stretches).all() or (stretches <= 0).any():
                return None  # Reject unrepresentable trials.
            packed = np.diag(stretches).ravel(order="F")
            value, full_J = force.eval(packed, need_jacobian=need_jacobian)
            residual = (value - target) / scale
            if not need_jacobian:
                return residual
            # Column-major diagonal entries 0, 4, 8; d(exp(p_j))/dp_j = exp(p_j).
            J = full_J[:, [0, 4, 8]].multiply(stretches[None, :] / scale).tocsc()
            return residual, J

    residual = PlasticResidual()

    def check_identifiability(p):
        singular = np.linalg.svd(residual.eval_jacobian(p).toarray(), compute_uv=False)
        ratio = singular[2] / singular[0]
        if not ratio > 1e-6:
            raise RuntimeError("Prescribed pose cannot locally identify all three plastic stretches")
        return ratio

    initial_p = np.log(initial)
    initial_ratio = check_identifiability(initial_p)
    truth_ratio = check_identifiability(np.log(truth))
    # Check the composed Jacobian, including the exponential parameterization.
    check_p = np.log(truth)
    J = residual.eval_jacobian(check_p).toarray()
    fd = np.empty_like(J)
    for j in range(3):
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
    print(f"truth={truth[0]:.12g} {truth[1]:.12g} {truth[2]:.12g}, "
          f"initial={initial[0]:.12g} {initial[1]:.12g} {initial[2]:.12g}, "
          f"fitted={fitted[0]:.12g} {fitted[1]:.12g} {fitted[2]:.12g}, converged={int(converged)}, "
          f"iterations={result.iterations}, parameter_error={parameter_error:.12g}, "
          f"force_residual={scale*relative_residual:.12g}, relative_residual={relative_residual:.12g}, "
          f"initial_ratio={initial_ratio:.12g}, truth_ratio={truth_ratio:.12g}, "
          f"fitted_ratio={fitted_ratio:.12g}, jacobian_error={jacobian_error:.12g}")
    return 0 if converged and parameter_error < 1e-7 and relative_residual < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
