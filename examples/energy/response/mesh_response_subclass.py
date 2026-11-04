"""Assemble subclass residuals and fit two shared parameters.

Run: uv run --no-sync python examples/energy/response/mesh_response_subclass.py
"""

import numpy as np
from litepgo import dof, energy, solver


class LinearResidual(energy.MeshResponse):
    def __init__(self, inputs, outputs, A, observed):
        super().__init__(inputs, outputs, batch_size=2)
        self.A, self.observed = A.copy(), observed.copy()

    def eval_batch(self, ids, x, need_jacobian):
        values = x @ self.A.T - self.observed[ids]
        return (values, np.tile(self.A, (len(ids), 1))) if need_jacobian else values

    def eval_jvp_batch(self, ids, x, dx):
        return dx @ self.A.T

    def eval_vjp_batch(self, ids, x, gy):
        return gy @ self.A


def main():
    inputs = dof.StencilDofMap(2, [[0, 1]] * 3, 1)
    outputs = dof.StencilDofMap(3, [[0, 1], [1, 2], [0, 2]], 1)
    A = np.array([[1.0, 2.0], [3.0, -1.0]])
    truth = np.array([2.0, 3.0])
    observed = np.tile(A @ truth, (3, 1))

    response = LinearResidual(inputs, outputs, A, observed)
    x = np.array([0.7, 1.2])
    # Products can be evaluated without constructing a global Jacobian.
    v, w = np.array([0.3, -0.8]), np.array([0.2, -0.5, 0.7])
    response.eval_jvp(x, v)
    response.eval_vjp(x, w)
    result = solver.gauss_newton_solve(
        response, x, solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    )
    if result.status != solver.GaussNewtonStatus.Converged:
        raise RuntimeError("Parameter fit failed")
    print("mesh response subclass: shared assembly, JVP/VJP and parameter fit")


if __name__ == "__main__":
    main()
