from litepgo.energy import AffineResponse
import numpy as np
import pytest
from litepgo import dof, fem, mesh


@pytest.mark.parametrize("basis,width", [(fem.HexQ1Basis, 3), (fem.HexHermiteBasis, 24)])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_hex_force_responses_match_energy_and_physical_derivatives(basis, width, order, batch_size):
    Q = fem.GaussLegendre3D[order]
    Energy = fem.VolumetricFemEnergy[basis, Q]
    Material = fem.VolumetricFemForceMaterialResponse[basis, Q]
    Plastic = fem.VolumetricFemForceFpResponse[basis, Q]
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]], dtype=np.int32)
    topology = mesh.HexTopology(12, cells)
    mapping = dof.VertexDofMap(topology, width)
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    Field = fem.InterpolatedResponse[fem.HexQ1Basis]
    geometry = []
    for e in range(2):
        X = (corners+[e, 0, 0])*[1.7, .8, 1.2]+[2., -.3, .4]
        if width == 3:
            X[:, 0] += .05*corners[:, 1]*corners[:, 2]
        geometry.append(Field(X))
    # Hermite includes nonzero physical derivatives and mixed-derivative slots.
    u = .002*np.sin(np.arange(12*width)+.3)
    p = np.array([[1.1, .05, 0], [.02, .95, .03], [0, .04, 1.03]]).ravel(order="F")
    theta = np.array([2., 3.])
    plastic = [AffineResponse(np.zeros((np.asarray(p).size, 3)), p) for _ in range(2)]
    parameters = [AffineResponse(np.zeros((np.asarray(theta).size, 3)), theta) for _ in range(2)]
    expected = Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, geometry, plastic, parameters).eval_gradient(u)
    material = Material(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(topology, 2), mapping, geometry, u, plastic, batch_size)
    fp = Plastic(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ConstantDofMap(topology, 9), mapping, geometry, u, parameters, batch_size)
    material_evaluation = material.evaluate(theta)
    M = material_evaluation.jacobian()
    g = material_evaluation.value
    fp_evaluation = fp.evaluate(p)
    P = fp_evaluation.jacobian()
    f = fp_evaluation.value
    np.testing.assert_allclose(g, expected, atol=1e-12)
    np.testing.assert_allclose(f, expected, atol=1e-12)
    np.testing.assert_allclose(M @ theta, expected, atol=1e-12)
    direction = np.linspace(-.2, .3, 9)
    def force_at(x):
        fields = [AffineResponse(np.zeros((np.asarray(x).size, 3)), x) for _ in range(2)]
        return Energy(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), mapping, geometry, fields, parameters).eval_gradient(u)
    np.testing.assert_allclose(P @ direction, (force_at(p+1e-6*direction)-force_at(p-1e-6*direction))/2e-6,
                               rtol=1e-6, atol=1e-8)
    me = Material(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ElementDofMap(topology, 2), mapping, geometry, u, plastic, batch_size)
    pe = Plastic(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), dof.ElementDofMap(topology, 9), mapping, geometry, u, parameters, batch_size)
    MJ = me.evaluate(np.tile(theta, 2)).jacobian().toarray()
    PJ = pe.evaluate(np.tile(p, 2)).jacobian().toarray()
    np.testing.assert_allclose(M.toarray(), MJ[:, :2]+MJ[:, 2:], atol=1e-13)
    np.testing.assert_allclose(P.toarray(), PJ[:, :9]+PJ[:, 9:], atol=1e-13)
    for response, point in [(material, theta), (fp, p), (me, np.tile(theta, 2)), (pe, np.tile(p, 2))]:
        derivative = response.evaluate(point).jacobian()
        d = np.sin(np.arange(response.num_inputs))
        w = np.cos(np.arange(response.num_outputs))
        np.testing.assert_allclose(response.evaluate(point).jvp(d), derivative @ d, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(response.evaluate(point).vjp(w), derivative.T @ w, rtol=1e-10, atol=1e-11)
    with pytest.raises(ValueError):
        fp.evaluate(np.zeros(9)).value
    np.testing.assert_allclose(fp.evaluate(p).value, expected, atol=1e-12)


@pytest.mark.parametrize("family,size", [(fem.VolumetricFemForceMaterialResponse, 2),
                                          (fem.VolumetricFemForceFpResponse, 9)])
def test_hermite_response_rejects_unsupported_geometry_and_wrong_map(family, size):
    cls = family[fem.HexHermiteBasis, fem.GaussLegendre3D[2]]
    topology = mesh.HexTopology(8, np.arange(8, dtype=np.int32).reshape(1, 8))
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    Field = fem.InterpolatedResponse[fem.HexQ1Basis]
    fixed = [AffineResponse(np.zeros((np.asarray(np.eye(3).ravel() if size == 2 else np.array([2., 3.])).size, 3)), np.eye(3).ravel() if size == 2 else np.array([2., 3.]))]
    params = dof.ConstantDofMap(topology, size)
    with pytest.raises(ValueError):
        cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, dof.VertexDofMap(topology, 3), [Field(corners)], np.zeros(24), fixed)
    sheared = corners.copy()
    sheared[:, 0] += .2*sheared[:, 1]
    with pytest.raises(ValueError):
        cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), params, dof.VertexDofMap(topology, 24), [Field(sheared)], np.zeros(192), fixed)
