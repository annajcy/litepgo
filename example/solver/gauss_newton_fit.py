"""Fit y(t) = amplitude * exp(-rate * t) with an analytic Jacobian."""

import numpy as np

from litepgo.solver import GaussNewtonOptions, GaussNewtonStatus, gauss_newton_solve


def main():
    t = np.linspace(0., 2., 9)
    observations = 2.5 * np.exp(-0.8 * t)

    def residual(parameters, need_jacobian):
        amplitude, rate = parameters
        decay = np.exp(-rate * t)
        r = amplitude * decay - observations
        # Trial points request residuals only; skip the Jacobian in that case.
        if not need_jacobian:
            return r
        return r, np.column_stack([decay, -amplitude * t * decay])

    result = gauss_newton_solve(residual, np.array([1., 0.3]), GaussNewtonOptions(gradient_tolerance=1e-10))
    assert result.status == GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, [2.5, 0.8], atol=1e-8, rtol=0)
    assert result.objective < 1e-16
    print(f"amplitude={result.x[0]:g}, rate={result.x[1]:g}, objective={result.objective:g}, iterations={result.iterations}")


if __name__ == "__main__":
    main()
