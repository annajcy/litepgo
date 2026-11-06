"""Fit three shared positive diagonal Fp stretches at a prescribed tetrahedral pose.

Python mirror of volume_fixed_pose_fp_fit.cpp. Geometry, displacement and material
parameters stay fixed; no equilibrium solve or plastic_matrix evolution is involved.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/fem/fitting/volume_fixed_pose_fp_fit.py
"""

from litepgo.energy import AffineResponse
from pathlib import Path

import numpy as np

from litepgo import dof, energy, fem, mesh, solver


class Residual(energy.FixedPatternResponse):
    def __init__(self, force, target, scale):
        super().__init__(np.ones((force.num_outputs, force.num_inputs)))
        self.force, self.target, self.scale = force, target, scale

    def eval_impl(self, p, need_jacobian):
        value, J = (self.force.eval_value_and_jacobian(p) if need_jacobian else (self.force.eval_value(p), None))
        residual = (value - self.target) / self.scale
        return (residual, J / self.scale) if need_jacobian else residual


def make_problem():
    volume = mesh.read_veg(Path(__file__).resolve().parents[3] / "assets/box-tet.veg")
    topology, nodes = volume.topology, volume.positions
    if topology.num_cells == 0:
        raise RuntimeError("Example requires a nonempty tetrahedral mesh")
    mapping = dof.VertexDofMap(topology, 3)
    XField = fem.InterpolatedResponse[fem.TetP1Basis]
    geometry = [
        XField(mapping.gather(e, nodes.ravel())) for e in range(mapping.num_elements)
    ]
    F = np.array([[1.12, 0.08, 0.02], [0.01, 0.96, 0.06], [0.03, 0.0, 1.07]])
    displacement = ((nodes - nodes.mean(axis=0)) @ (F - np.eye(3)).T).ravel()
    truth, initial = np.array([1.08, 0.94, 1.03]), np.ones(3)
    theta = [AffineResponse(np.zeros((2, 3)), np.array([2.0, 3.0])) for _ in geometry]
    ElasticEnergy = fem.VolumetricFemEnergy[
        fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
    ]
    elastic = ElasticEnergy(
        mapping,
        geometry,
        [AffineResponse(np.zeros((9, 3)), np.diag(truth).ravel(order="F")) for _ in geometry],
        theta,
    )
    # Target comes from the energy gradient, independently of the force-response path.
    target = elastic.eval_gradient(displacement)
    scale = np.linalg.norm(target)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Calibration requires finite nonzero target forces")
    ForceParameterResponse = fem.VolumetricFemForceFpResponse[
        fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature
    ]
    force = ForceParameterResponse(
        dof.ConstantDofMap(topology, 9), mapping, geometry, displacement, theta
    )

    parameterized_force = energy.ComposedResponse(force, fem.LogDiagonalFpResponse(3))
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
        f"truth={truth[0]:.12g} {truth[1]:.12g} {truth[2]:.12g}, "
        f"initial={initial[0]:.12g} {initial[1]:.12g} {initial[2]:.12g}, "
        f"fitted={fitted[0]:.12g} {fitted[1]:.12g} {fitted[2]:.12g}, converged={int(converged)}, "
        f"iterations={result.iterations}, parameter_error={parameter_error:.12g}, "
        f"force_residual={scale * relative_residual:.12g}, relative_residual={relative_residual:.12g}"
    )
    return 0 if converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
