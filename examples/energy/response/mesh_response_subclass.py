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
    A = np.array([[1., 2.], [3., -1.]])
    truth = np.array([2., 3.])
    observed = np.tile(A @ truth, (3, 1))

    response = LinearResidual(inputs, outputs, A, observed)
    x = np.array([.7, 1.2])
    # Independently sum the local rows assigned to each global output.
    J = np.array([[2., 4.], [4., 1.], [6., -2.]])
    np.testing.assert_allclose(response.eval_value(x), J @ (x - truth))
    np.testing.assert_allclose(response.eval_jacobian(x).toarray(), J)
    v, w = np.array([.3, -.8]), np.array([.2, -.5, .7])
    jv, jtw = response.eval_jvp(x, v), response.eval_vjp(x, w)
    np.testing.assert_allclose(jv, J @ v)
    np.testing.assert_allclose(jtw, J.T @ w)
    np.testing.assert_allclose(w @ jv, v @ jtw)
    h = 1e-5
    np.testing.assert_allclose(
        jv, (response.eval_value(x + h*v) - response.eval_value(x - h*v))/(2*h),
        atol=1e-9,
    )
    result = solver.gauss_newton_solve(
        response, x, solver.GaussNewtonOptions(gradient_tolerance=1e-10)
    )
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, truth, atol=1e-8, rtol=0)
    print("mesh response subclass: shared assembly, JVP/VJP and parameter fit verified")


if __name__ == "__main__":
    main()
