import scipy.sparse
from litepgo.energy import AffineResponse
import gc
import numpy as np
import pytest
from litepgo import dof, fem, mesh, energy, solver
from test_material_force_response import problem, Energy

Force = fem.VolumetricFemForceFpResponse[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]


@pytest.mark.parametrize("batch_size", [1, 3, 256])
def test_fp_force_energy_jacobian_and_recovery(batch_size):
    topology, mapping, geometry, u, plastic = problem()
    theta = [AffineResponse(np.zeros((np.asarray(np.array([2., 3.])).size, 3)), np.array([2., 3.])), AffineResponse(np.zeros((np.asarray(np.array([4., 2.])).size, 3)), np.array([4., 2.]))]
    p = np.concatenate([field.eval_value(np.zeros(3)) for field in plastic])
    obj = Force(dof.ElementDofMap(topology, 9), mapping, geometry, u, theta=theta, batch_size=batch_size)
    assert isinstance(obj, energy.MeshResponse)
    expected = Energy(mapping, geometry, plastic, theta).eval_gradient(u)
    value, J = obj.eval_value_and_jacobian(p)
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




def test_fp_force_fits_a_constrained_plastic_scale():
    topology, mapping, geometry, u, _ = problem()
    theta = [AffineResponse(np.zeros((np.asarray(np.array([2., 3.])).size, 3)), np.array([2., 3.])) for _ in range(2)]
    obj = Force(dof.ConstantDofMap(topology, 9), mapping, geometry, u, theta)
    direction = np.eye(3).ravel(order="F")
    target = obj.eval_value(1.05*direction)

    class ScaleResidual(energy.FixedPatternResponse):
        def __init__(self):
            import scipy.sparse
            super().__init__(scipy.sparse.csc_matrix(np.ones((15, 1))))

        def eval_impl(self, x, need):
            value, J = (obj.eval_value_and_jacobian(x[0]*direction) if need else (obj.eval_value(x[0]*direction), None))
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
    theta = [AffineResponse(np.zeros((np.asarray(np.array([2., 3.])).size, 3)), np.array([2., 3.])) for _ in range(2)]
    for bad_theta in ([], [AffineResponse(np.zeros((np.asarray(np.ones(9)).size, 3)), np.ones(9))]*2):
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
