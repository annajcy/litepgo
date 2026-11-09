"""Fixed reference points, variable coefficients, and reference-derivative sampling."""
import numpy as np
from litepgo import dof, energy, fem, mesh


def main():
    B = fem.HexQ1Basis
    points = np.array([[.2, .3, .4], [.6, .7, .8]])
    # Two components: f(xi)=xi_1*xi_2, g(xi)=xi_2*xi_3.
    C = np.array([[p[0]*p[1], p[1]*p[2]]
                  for p in (B.coefficient_info(a).point for a in range(B.num_basis_functions))])
    c = C.ravel()  # basis coefficient first, then component
    values = fem.value_sampling_response[B](points, components=2)
    gradients = fem.gradient_sampling_response[B](points, components=2)
    hessians = fem.hessian_sampling_response[B](points, components=2)
    assert all(type(r) is energy.AffineResponse for r in (values, gradients, hessians))
    np.testing.assert_allclose(values.evaluate(c).value.reshape(2, 2), [[0.06, 0.12], [0.42, 0.56]])
    # Per point: xi direction first, then component.
    np.testing.assert_allclose(gradients.evaluate(c).value.reshape(2, 3, 2), [[[0.3, 0], [0.2, 0.4], [0, 0.3]], [[0.7, 0], [0.6, 0.8], [0, 0.7]]])
    # Per point: (11,12,13,22,23,33), then component; no factor of two.
    expected = np.array([[0, 0], [1, 0], [0, 0], [0, 0], [0, 1], [0, 0]])
    np.testing.assert_allclose(hessians.evaluate(c).value.reshape(2, 6, 2), np.tile(expected, (2, 1, 1)))
    A = gradients.evaluate(c).jacobian()
    direction = np.ones_like(c)
    weights = np.arange(gradients.num_outputs, dtype=float)
    np.testing.assert_allclose(gradients.evaluate(c).jvp(direction), A @ direction)
    np.testing.assert_allclose(gradients.evaluate(c).vjp(weights), A.T @ weights)
    # The sampled spatial Hessian is nonzero; its derivative twice with respect to c is zero.
    assert hessians.evaluate(c).hessian(np.ones(hessians.num_outputs)).nnz == 0
    geometry = energy.AffineResponse(np.diag([2., 3., 4.]))
    deformation = fem.deformation_gradient_sampling_response[B](geometry, points)
    U = .1 * np.array([B.coefficient_info(a).point for a in range(8)])
    packed_F = deformation.evaluate(U.ravel()).value.reshape(-1, 9)
    expected_F = np.diag([1.05, 1 + .1 / 3, 1.025])
    for packed in packed_F:
        np.testing.assert_allclose(packed.reshape(3, 3, order="F"), expected_F, atol=1e-14)
    # Compose the FEM calculation explicitly: fixed samples, material, integration.
    Q = fem.GaussLegendre3D[2]
    element = fem.VolumetricElement3D[Q](geometry)
    sample = fem.deformation_gradient_sampling_response[B](geometry, element.points())
    coefficients = U.ravel()  # Q1 nodal DOFs already equal reference coefficients.
    A = sample.evaluate(coefficients).jacobian().toarray()
    samples = sample.evaluate(coefficients).value.reshape(-1, 9)
    Fp, theta = np.eye(3), np.array([2., 3.])
    E, g, H = 0., np.zeros(24), np.zeros((24, 24))
    for q, weight in enumerate(element.weights()):
        F = samples[q].reshape(3, 3, order="F")
        Bq = A[9*q:9*q+9]
        P = fem.NeoHookean.eval_dW_dF(F, Fp, theta).ravel(order="F")
        tangent = fem.NeoHookean.eval_d2W_dF2(F, Fp, theta)
        E += weight * fem.NeoHookean.eval_W(F, Fp, theta)
        g += weight * Bq.T @ P
        H += weight * Bq.T @ tangent @ Bq
    topology = mesh.HexTopology(8, np.arange(8, dtype=np.int32)[None])
    mapping = dof.VertexDofMap(topology, 3)
    fixed_Fp = energy.AffineResponse(np.zeros((9, 3)), Fp.ravel(order="F"))
    fixed_theta = energy.AffineResponse(np.zeros((2, 3)), theta)
    native = fem.VolumetricFemEnergy[B, Q](fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, [geometry], [fixed_Fp], [fixed_theta])
    np.testing.assert_allclose(E, native.eval_value(coefficients), atol=1e-13)
    np.testing.assert_allclose(g, native.eval_gradient(coefficients), atol=1e-13)
    np.testing.assert_allclose(H, native.eval_hessian(coefficients).toarray(), atol=1e-13)
    print("coefficient value, gradient and Hessian sampling: verified")


if __name__ == "__main__":
    main()
