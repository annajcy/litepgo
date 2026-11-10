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
        super().__init__(np.ones((force.num_outputs, force.num_inputs)))
        self.force, self.target, self.scale = force, target, scale

    def eval_impl(self, p, need_jacobian):
        evaluation = self.force.evaluate(p)
        J = evaluation.jacobian() if need_jacobian else None
        value = evaluation.value
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
    # The energy fixes Fp, while the force response below makes it the active input.
    target_energy = fem.ShellFemEnergy(fem.PointwiseShellAdapter[fem.KoiterStVK](),
        mapping, rest, thickness, theta, [true_Fp] * mapping.num_elements
    )
    target = target_energy.evaluate(displacement).gradient()
    scale = np.linalg.norm(target)
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Calibration requires finite nonzero target forces")
    force = fem.ShellFemForceFpResponse(fem.PointwiseShellAdapter[fem.KoiterStVK](),
        dof.ConstantDofMap(topology, 9), mapping, rest, displacement, thickness, theta
    )

    # Fixed pose and original face areas are cached; only the parameter vector varies.
    np.testing.assert_allclose(force.evaluate(true_Fp.ravel(order='F')).value, target, rtol=1e-11, atol=1e-12)
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
    relative_residual = np.linalg.norm(residual.evaluate(result.x).value)
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
