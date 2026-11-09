"""Fit the parameters of a shell material written in PyTorch from equilibrium reactions.

The Koiter density below is written in Torch and passed as a material instance to
native shell FEM; the problem, measurement and Gauss-Newton fit are those of
shell_equilibrium_material_fit.py. The fit is repeated with the native KoiterStVK.
Run: uv run --no-sync python examples/fem/fitting/torch_shell_material_fit.py
"""

from functools import partial

import numpy as np
import torch
from litepgo import fem, solver
from litepgo.torch import TorchShellMaterial
from shell_equilibrium_material_fit import Residual, make_factories, make_problem, measure


def unpack(packed):
    """Packed (s11, s12, s22) columns to symmetric (N,2,2) matrices."""
    return torch.stack([packed[:, 0], packed[:, 1], packed[:, 1], packed[:, 2]], -1).reshape(-1, 2, 2)


class TorchKoiter(TorchShellMaterial):
    """W = t psi((a - a_bar)/2) + t^3/12 psi(b - b_bar), theta = [lambda_ps, mu].

    psi(S) = lambda/2 tr(a_bar^-1 S)^2 + mu tr(a_bar^-1 S a_bar^-1 S), as in KoiterStVK.
    """

    num_parameters = 2

    @staticmethod
    def W(ab, ab_bar, thickness, theta):
        a, b = unpack(ab[:, :3]), unpack(ab[:, 3:])
        a_bar, b_bar = unpack(ab_bar[:, :3]), unpack(ab_bar[:, 3:])
        inverse = torch.linalg.inv(a_bar)
        lam, mu = theta.unbind(-1)

        def psi(S):
            M = inverse @ S
            trace = M.diagonal(dim1=-2, dim2=-1).sum(-1)
            return 0.5 * lam * trace.square() + mu * (M @ M).diagonal(dim1=-2, dim2=-1).sum(-1)

        return thickness * psi(0.5 * (a - a_bar)) + thickness**3 / 12 * psi(b - b_bar)


def fit(material):
    problem = make_problem()
    measurement = partial(measure, problem, *make_factories(problem, material))
    truth, initial = np.array([2.0, 3.0]), np.array([0.7, 1.2])
    target = measurement(truth).reaction
    residual = Residual(measurement, target, np.linalg.norm(target))
    result = solver.gauss_newton_solve(
        residual, initial, solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    )
    if result.status != solver.GaussNewtonStatus.Converged:
        raise RuntimeError(f"{type(material).__name__} fit did not converge")
    return result.x, np.linalg.norm(result.x - truth) / np.linalg.norm(truth)


def main():
    torch_fit, torch_error = fit(TorchKoiter())
    native_fit, native_error = fit(fem.PointwiseShellAdapter[fem.KoiterStVK]())
    print(
        f"torch: fitted={torch_fit[0]:.12g} {torch_fit[1]:.12g}, parameter_error={torch_error:.3e}; "
        f"native: parameter_error={native_error:.3e}; "
        f"fit_difference={np.abs(torch_fit - native_fit).max():.3e}"
    )


if __name__ == "__main__":
    main()
