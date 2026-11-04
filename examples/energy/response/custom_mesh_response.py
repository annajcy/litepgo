"""Assemble callback residuals and fit two shared parameters.

Run: uv run --no-sync python examples/energy/response/custom_mesh_response.py
"""
import numpy as np
from litepgo import dof, energy, solver


def main():
    inputs = dof.StencilDofMap(2, [[0, 1]] * 3, 1)
    outputs = dof.StencilDofMap(3, [[0, 1], [1, 2], [0, 2]], 1)
    A = np.array([[1., 2.], [3., -1.]])
    truth = np.array([2., 3.])
    observed = np.tile(A @ truth, (3, 1))

    def batch(ids, x, need_jacobian):
        values = x @ A.T - observed[ids]
        return (values, np.tile(A, (len(ids), 1))) if need_jacobian else values

    response = energy.CustomMeshResponse(inputs, outputs, batch, batch_size=2)
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
    print("custom mesh response: shared assembly, JVP/VJP and parameter fit verified")


if __name__ == "__main__":
    main()
