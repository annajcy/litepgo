import gc
import numpy as np
import pytest
from litepgo import dof, fem, mesh, response, solver
from test_material_force_response import problem, Energy

Force = fem.VolumetricFemForceFpResponse[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]


@pytest.mark.parametrize("batch_size", [1, 3, 256])
def test_fp_force_energy_jacobian_and_recovery(batch_size):
    topology, mapping, geometry, u, plastic = problem()
    theta = [fem.IdentityField3D(np.array([2., 3.])), fem.IdentityField3D(np.array([4., 2.]))]
    p = np.concatenate([field.eval_value(np.zeros(3)) for field in plastic])
    obj = Force(dof.ElementDofMap(topology, 9), mapping, geometry, u, theta=theta, batch_size=batch_size)
    assert isinstance(obj, response.Response)
    expected = Energy(mapping, geometry, plastic, theta).eval_gradient(u)
    value, J = obj.eval(p, need_jacobian=True)
    np.testing.assert_allclose(value, expected, atol=1e-13)
    for c in range(18):
        delta = np.zeros(18)
        delta[c] = 1e-6
        np.testing.assert_allclose(J.toarray()[:, c], (obj.eval_value(p+delta)-obj.eval_value(p-delta))/2e-6,
                                   rtol=1e-7, atol=1e-9)
    shared = Force(dof.ConstantDofMap(topology, 9), mapping, geometry, u, theta)
    repeated = np.tile(p[:9], 2)
    element_J = obj.eval_jacobian(repeated).toarray()
    np.testing.assert_allclose(shared.eval_jacobian(p[:9]).toarray(), element_J[:, :9]+element_J[:, 9:], atol=1e-13)
    with pytest.raises(ValueError):
        obj.eval_value(np.zeros(18))
    np.testing.assert_array_equal(obj.eval_jacobian(p).indices, J.indices)
    u[:] = 0
    del mapping, geometry, theta, topology
    gc.collect()
    np.testing.assert_allclose(obj.eval_value(p), expected, atol=1e-13)


def test_local_plastic_chain_rule_lifetime_and_material_api():
    Local = fem.VolumetricFpForceLocalResponse3D[fem.NeoHookean]
    F = np.array([[1.1, .1, 0], [0, .9, .1], [.02, 0, 1.2]])
    p = np.eye(3)
    theta = np.array([2., 3.])
    local = Local(fem.IdentityField3D(F.ravel(order="F")), fem.IdentityField3D(p.ravel(order="F")),
                  fem.IdentityField3D(theta))
    gc.collect()
    B = fem.NeoHookean.eval_d2W_dF_dFp(F, p, theta)
    np.testing.assert_allclose(local.eval_jacobian(np.zeros(3)), B)
    value, J = local.accumulate(np.zeros(3), .4, jacobian=True)
    np.testing.assert_allclose(value, .4*fem.NeoHookean.eval_dW_dF(F, p, theta).ravel(order="F"))
    np.testing.assert_allclose(J, .4*B)
    with pytest.raises(ValueError):
        fem.NeoHookean.eval_d2W_dF_dFp(F, np.zeros((3, 3)), theta)


def test_fp_force_fits_a_constrained_plastic_scale():
    topology, mapping, geometry, u, _ = problem()
    theta = [fem.IdentityField3D(np.array([2., 3.])) for _ in range(2)]
    obj = Force(dof.ConstantDofMap(topology, 9), mapping, geometry, u, theta)
    direction = np.eye(3).ravel(order="F")
    target = obj.eval_value(1.05*direction)

    class ScaleResidual(response.FixedPatternResponse):
        def __init__(self):
            import scipy.sparse
            super().__init__(scipy.sparse.csc_matrix(np.ones((15, 1))))

        def eval_values(self, x, need):
            value, J = obj.eval(x[0]*direction, need_jacobian=need)
            if not need:
                return value-target
            import scipy.sparse
            return value-target, scipy.sparse.csc_matrix((J @ direction).reshape(-1, 1))

    result = solver.gauss_newton_solve(ScaleResidual(), np.array([1.]),
                                     solver.GaussNewtonOptions(gradient_tolerance=1e-10))
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, [1.05], atol=1e-8)


def test_fp_force_validation_and_empty_mesh():
    topology, mapping, geometry, u, _ = problem()
    params = dof.ConstantDofMap(topology, 9)
    theta = [fem.IdentityField3D(np.array([2., 3.])) for _ in range(2)]
    for bad_theta in ([], [fem.IdentityField3D(np.ones(9))]*2):
        with pytest.raises(ValueError):
            Force(params, mapping, geometry, u, bad_theta)
    with pytest.raises(ValueError):
        Force(params, mapping, geometry, u, theta, batch_size=0)
    with pytest.raises(ValueError):
        Force(dof.ConstantDofMap(topology, 2), mapping, geometry, u, theta)
    empty = mesh.TetTopology(0, np.empty((0, 4), dtype=np.int32))
    zero = Force(dof.ConstantDofMap(empty, 9), dof.VertexDofMap(empty, 3), [], np.empty(0), [])
    assert zero.eval_value(np.eye(3).ravel()).size == 0
    assert zero.eval_jacobian(np.eye(3).ravel()).shape == (0, 9)
