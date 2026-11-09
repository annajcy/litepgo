"""Subclass fixed/dynamic Response and pass the residual object to Gauss-Newton.

Run from the repository root after building litepgo:
    uv run --no-sync python examples/energy/response/custom_response.py
"""

import numpy as np
from litepgo import energy, solver


class LinearResidual(energy.FixedPatternResponse):
    def __init__(self):
        self.J = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        super().__init__(self.J)

    def eval_impl(self, x, need_jacobian):
        r = self.J @ x - [1.0, 2.0, 3.0]
        return (r, self.J) if need_jacobian else r


class NonlinearResidual(energy.DynamicPatternResponse):
    def __init__(self):
        super().__init__(2, 2)

    def jacobian_pattern_at(self, x):
        return np.array([[1.0, 0.0], [float(x[0] > 0), 1.0]])

    def eval_impl(self, x, need_jacobian):
        s = max(0.0, x[0])
        r = np.array([x[0] - 1, x[1] + s * s - 2])
        return (r, np.array([[1.0, 0.0], [2 * s, 1.0]])) if need_jacobian else r


def main():
    fixed, dynamic = LinearResidual(), NonlinearResidual()
    fixed_evaluation = fixed.evaluate(np.array([1.0, 2.0]))
    J = fixed_evaluation.jacobian()
    r = fixed_evaluation.value
    active = np.ones(2)
    dynamic.evaluate(active).jacobian()
    dynamic.evaluate(-active).jacobian()
    result = solver.gauss_newton_solve(
        dynamic, -active, solver.GaussNewtonOptions(gradient_tolerance=1e-9)
    )
    if result.status != solver.GaussNewtonStatus.Converged:
        raise RuntimeError("Response fit failed")
    print(f'fixed Jacobian={J.shape[0]}x{J.shape[1]}, dynamic solution={result.x[0]:g} {result.x[1]:g}, Jacobian nnz={dynamic.evaluate(result.x).jacobian().nnz}')


if __name__ == "__main__":
    main()
