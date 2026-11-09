import gc
import weakref

import numpy as np
import pytest
from litepgo import dof, energy, fem, mesh


@pytest.mark.parametrize("basis", [fem.TetP1Basis, fem.HexQ1Basis, fem.HexHermiteBasis])
def test_spatial_deformation_jacobian_and_coefficient_snapshots(basis):
    G = fem.TetP1Basis if basis is fem.TetP1Basis else fem.HexQ1Basis
    nodes = np.array([G.coefficient_info(i).point for i in range(G.num_basis_functions)])
    nodes *= [2., 3., 4.]
    nodes[:, 0] += .07 * nodes[:, 1] * nodes[:, 2]
    rng = np.random.default_rng(7)
    c = rng.normal(size=(basis.num_basis_functions, 3)) * .01
    geometry = fem.InterpolatedResponse[G](nodes)
    displacement = fem.InterpolatedResponse[basis](c)
    response = fem.DeformationGradientResponse(displacement, geometry)
    xi = np.array([.2, .25, .3])
    expected = np.eye(3) + (c.T @ basis.eval_dN_dxi(xi)) @ np.linalg.inv(nodes.T @ G.eval_dN_dxi(xi))
    np.testing.assert_allclose(response.evaluate(xi).value, expected.ravel(order='F'))
    c[:] = 0
    nodes[:] = 0
    h = 1e-6
    fd = np.column_stack([(response.evaluate(xi + h * d).value - response.evaluate(xi - h * d).value) / (2 * h) for d in np.eye(3)])
    np.testing.assert_allclose(response.evaluate(xi).jacobian().toarray(), fd, atol=2e-10)
    assert not response.has_hessian
    with pytest.raises(RuntimeError):
        response.evaluate(xi).hessian(np.ones(9))
    del geometry, displacement
    gc.collect()
    np.testing.assert_allclose(response.evaluate(xi).value, expected.ravel(order='F'))


class SpatialConstant(energy.FixedPatternResponse):
    def __init__(self, value):
        super().__init__(np.zeros((len(value), 3)))
        self.value = np.array(value, dtype=float)
        self.queries = 0

    def eval_impl(self, xi, need_jacobian):
        self.queries += 1
        if need_jacobian:
            raise AssertionError("fixed material inputs need only values")
        return self.value


def test_fem_snapshots_material_responses_and_releases_owners():
    nodes = np.vstack([np.zeros(3), np.eye(3)])
    topology = mesh.TetTopology(4, np.array([[0, 1, 2, 3]], dtype=np.int32))
    mapping = dof.VertexDofMap(topology, 3)
    geometry = fem.InterpolatedResponse[fem.TetP1Basis](nodes)
    plastic = SpatialConstant(np.eye(3).ravel())
    material = SpatialConstant([2., 3.])
    cls = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]
    elastic = cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, [geometry], [plastic], [material])
    force_cls = fem.VolumetricFemForceMaterialResponse[fem.TetP1Basis, fem.TetCentroidQuadrature]
    u = .1 * nodes.ravel()
    force = force_cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(topology, 2), mapping, [geometry], u, [plastic])
    value, g, H = elastic.eval(u, gradient=True, hessian=energy.HessianKind.Exact)
    assert (plastic.queries, material.queries) == (2, 1)
    material.value *= 2
    plastic.value *= 1.1
    assert elastic.eval_value(u) == value
    rebuilt = cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, [geometry], [SpatialConstant(np.eye(3).ravel())], [material])
    np.testing.assert_allclose(rebuilt.eval_gradient(u), 2*g)
    owners = [weakref.ref(obj) for obj in [geometry, plastic, material]]
    del geometry, plastic, material
    gc.collect()
    assert all(ref() is None for ref in owners)
    np.testing.assert_allclose(force.evaluate(np.array([2.0, 3.0])).value, g)
    np.testing.assert_allclose(elastic.eval_hessian(u).toarray(), H.toarray())
    with pytest.raises((TypeError, ValueError)):
        cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, [None], [], [])


def test_spatial_deformation_requires_valid_geometry_and_child_capabilities():
    identity = energy.AffineResponse(np.eye(3))
    zero = energy.AffineResponse(np.zeros((3, 3)))
    for invalid in [energy.AffineResponse(np.zeros((2, 3))), energy.AffineResponse(np.zeros((3, 2)))]:
        with pytest.raises(ValueError):
            fem.DeformationGradientResponse(identity, invalid)
    for matrix in [np.zeros((3, 3)), np.diag([-1., 1., 1.])]:
        r = fem.DeformationGradientResponse(zero, energy.AffineResponse(matrix))
        with pytest.raises(ValueError):
            r.evaluate(np.zeros(3)).value
    r = fem.DeformationGradientResponse(zero, identity)
    np.testing.assert_array_equal(r.evaluate(np.zeros(3)).value, np.eye(3).ravel())
    np.testing.assert_array_equal(r.evaluate(np.zeros(3)).jacobian().toarray(), np.zeros((9, 3)))
    with pytest.raises(ValueError):
        fem.DeformationGradientResponse(r, identity)


def test_element_step_callback_uses_every_quadrature_point_and_validates_bounds():
    Q = fem.GaussLegendre3D[2]
    element = fem.VolumetricElement3D[Q](energy.AffineResponse(np.eye(3)))
    visited = []

    def step(xi):
        visited.append(xi.copy())
        return .2 + .3 * xi[2]

    assert element.max_feasible_step(step) == pytest.approx(.2 + .3 * Q.points()[:, 2].min())
    np.testing.assert_array_equal(visited, Q.points())
    for bound in [-.1, 1.1, np.nan]:
        with pytest.raises(RuntimeError):
            element.max_feasible_step(lambda xi: bound)


@pytest.mark.parametrize("basis", [fem.TetP1Basis, fem.HexQ1Basis])
def test_locator_rejects_arbitrary_response_without_supported_geometry_bounds(basis):
    with pytest.raises(ValueError):
        fem.ElementLocator[basis]([energy.AffineResponse(np.eye(3))])


def test_element_exposes_independent_reference_points_and_physical_weights():
    Q = fem.GaussLegendre3D[2]
    geometry = energy.AffineResponse(np.diag([2., 3., 4.]))
    owner = weakref.ref(geometry)
    element = fem.VolumetricElement3D[Q](geometry)
    del geometry
    gc.collect()
    assert owner() is None
    points, weights = element.points(), element.weights()
    np.testing.assert_array_equal(points, Q.points())
    np.testing.assert_allclose(weights, 24 * Q.weights())
    assert weights.sum() == pytest.approx(24)
    points[:] = 0
    weights[:] = 0
    np.testing.assert_array_equal(element.points(), Q.points())
    np.testing.assert_allclose(element.weights(), 24 * Q.weights())
