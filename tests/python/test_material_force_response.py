from litepgo.energy import AffineResponse
import gc
import numpy as np
import pytest
from litepgo import dof, fem, mesh, energy, solver

Force = fem.VolumetricFemForceMaterialResponse[fem.TetP1Basis, fem.TetCentroidQuadrature]
Energy = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]


def problem():
    nodes = np.array([[0., 0., 0.], [1.2, 0., 0.], [0., 1., 0.], [0., 0., .8], [1.2, 1., .8]])
    topology = mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))
    mapping = dof.VertexDofMap(topology, 3)
    Field = fem.InterpolatedResponse[fem.TetP1Basis]
    geometry = [Field(mapping.gather(e, nodes.ravel())) for e in range(2)]
    F = np.array([[1.15, .12, 0.], [.03, .94, .08], [0., .02, 1.07]])
    u = (nodes @ (F - np.eye(3)).T).ravel()
    plastic = [AffineResponse(np.zeros((np.asarray(np.array([[1.1+.05*e, 0., 0.], [0., 1., .1], [0., 0., 1.]]).ravel(order="F")).size, 3)), np.array([[1.1+.05*e, 0., 0.], [0., 1., .1], [0., 0., 1.]]).ravel(order="F")) for e in range(2)]
    return topology, mapping, geometry, u, plastic


def observed(data, theta):
    _, mapping, geometry, u, plastic = data
    fields = [AffineResponse(np.zeros((np.asarray(theta if theta.size == 2 else theta[2*e:2*e+2]).size, 3)), theta if theta.size == 2 else theta[2*e:2*e+2]) for e in range(2)]
    return Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, geometry, plastic, fields).eval_gradient(u)


@pytest.mark.parametrize("batch_size", [1, 3, 256])
def test_shared_and_element_parameters_match_energy(batch_size):
    data = problem()
    topology, mapping, geometry, u, plastic = data
    shared = Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(topology, 2), mapping, geometry, u, plastic, batch_size)
    element = Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ElementDofMap(topology, 2), mapping, geometry, u, plastic, batch_size)
    theta = np.array([2., 3.])
    independent = np.array([2., 3., 5., .8])
    assert isinstance(shared, energy.Response)
    assert isinstance(shared, fem.VolumetricFemForceMaterialResponse)
    assert isinstance(shared, energy.MeshResponse)
    shared_evaluation = shared.evaluate(theta)
    J = shared_evaluation.jacobian()
    force = shared_evaluation.value
    element_evaluation = element.evaluate(independent)
    E = element_evaluation.jacobian()
    per_force = element_evaluation.value
    np.testing.assert_allclose(force, observed(data, theta), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(per_force, observed(data, independent), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(J.toarray(), E[:, :2].toarray() + E[:, 2:].toarray(), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(E @ independent, per_force, rtol=1e-12, atol=1e-13)
    for response, point in [(shared, theta), (element, independent)]:
        derivative = response.evaluate(point).jacobian()
        d = np.sin(np.arange(response.num_inputs))
        w = np.cos(np.arange(response.num_outputs))
        np.testing.assert_allclose(response.evaluate(point).jvp(d), derivative @ d, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(response.evaluate(point).vjp(w), derivative.T @ w, rtol=1e-10, atol=1e-11)
    with pytest.raises(ValueError):
        shared.evaluate(np.array([2.0, 0.0])).value
    np.testing.assert_allclose(shared.evaluate(theta).value, force)
    np.testing.assert_array_equal(shared.evaluate(theta).jacobian().indices, J.indices)
    # Input displacement is copied; temporary maps/field wrappers remain alive.
    u[:] = 0
    del topology, mapping, geometry, plastic, data
    gc.collect()
    np.testing.assert_allclose(shared.evaluate(theta).value, force)


def test_parameter_recovery_and_batch_consistency():
    data = problem()
    topology, mapping, geometry, u, plastic = data
    theta = np.array([2., 3.])
    target = observed(data, theta)
    r1 = Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(topology, 2), mapping, geometry, u, plastic, 1)
    r2 = Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(topology, 2), mapping, geometry, u, plastic, 3)
    np.testing.assert_array_equal(r1.evaluate(theta).jacobian().data, r2.evaluate(theta).jacobian().data)
    class Residual(energy.FixedPatternResponse):
        def __init__(self):
            super().__init__(np.ones((r2.num_outputs, r2.num_inputs)))
        def eval_impl(self, x, need):
            evaluation = r2.evaluate(x)
            J = evaluation.jacobian() if need else None
            value = evaluation.value
            return (value-target, J) if need else value-target
    result = solver.gauss_newton_solve(Residual(), np.array([.7, 1.2]), solver.GaussNewtonOptions(gradient_tolerance=1e-10))
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, theta, atol=1e-7, rtol=0)


def test_validation_and_empty_mesh():
    topology, mapping, geometry, u, plastic = problem()
    params = dof.ConstantDofMap(topology, 2)
    with pytest.raises(ValueError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, mapping, geometry, u[:-1], plastic)
    with pytest.raises(ValueError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, mapping, [], u, plastic)
    with pytest.raises(ValueError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, mapping, geometry, u, [])
    with pytest.raises(ValueError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, mapping, geometry, u, plastic, 0)
    with pytest.raises(ValueError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(topology, 3), mapping, geometry, u, plastic)
    with pytest.raises(ValueError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, mapping, geometry, np.full(15, np.nan), plastic)
    with pytest.raises(ValueError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, mapping, geometry, u, [AffineResponse(np.zeros((np.asarray(np.ones(8)).size, 3)), np.ones(8))]*2)
    with pytest.raises(TypeError):
        Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.VertexDofMap(topology, 2), mapping, geometry, u, plastic)
    empty = mesh.TetTopology(0, np.empty((0, 4), dtype=np.int32))
    zero = Force(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(empty, 2), dof.VertexDofMap(empty, 3), [], np.empty(0), [])
    assert zero.evaluate(np.array([2.0, 3.0])).value.size == 0
    assert zero.evaluate(np.array([2.0, 3.0])).jacobian().shape == (0, 2)
