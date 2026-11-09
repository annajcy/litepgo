import gc
import weakref
import numpy as np
import pytest
from litepgo import energy, fem

@pytest.mark.parametrize('B', [fem.TetP1Basis, fem.HexQ1Basis, fem.HexHermiteBasis])
def test_coefficient_map_matches_spatial_query_and_finite_differences(B):
    G = fem.TetP1Basis if B is fem.TetP1Basis else fem.HexQ1Basis
    X = np.array([G.coefficient_info(a).point for a in range(G.num_basis_functions)]) * [2,3,4]
    X[:,0] += .03 * X[:,1] * X[:,2]
    geometry = fem.InterpolatedResponse[G](X)
    points = np.array([[.1,.2,.3], [.3,.1,.2]])
    C = np.random.default_rng(3).normal(size=(B.num_basis_functions,3))*.01
    c = C.ravel()
    sampling = fem.deformation_gradient_sampling_response[B](geometry, points)
    assert type(sampling) is energy.AffineResponse
    spatial = fem.DeformationGradientResponse(fem.InterpolatedResponse[B](C), geometry)
    for xi in points:
        J_spatial = spatial.eval_jacobian(xi)
        d, w = np.array([.2, -.3, .4]), np.linspace(-.5, .7, 9)
        np.testing.assert_allclose(spatial.eval_jvp(xi, d), J_spatial @ d, atol=1e-13)
        np.testing.assert_allclose(spatial.eval_vjp(xi, w), J_spatial.T @ w, atol=1e-13)
    expected = np.concatenate([spatial.eval_value(xi) for xi in points])
    np.testing.assert_allclose(sampling.eval_value(c), expected, atol=1e-14)
    J = sampling.eval_jacobian(c).toarray()
    h = 1e-6
    fd = np.column_stack([(sampling.eval_value(c+h*d)-sampling.eval_value(c-h*d))/(2*h) for d in np.eye(len(c))])
    np.testing.assert_allclose(J, fd, atol=2e-10)
    assert sampling.eval_hessian(c, np.ones(18)).nnz == 0
    ref = weakref.ref(geometry)
    del spatial, geometry
    gc.collect()
    assert ref() is None
    points[:] = 0
    np.testing.assert_allclose(sampling.eval_value(c), expected, atol=1e-14)

def test_empty_and_invalid_sampling():
    factory = fem.deformation_gradient_sampling_response[fem.TetP1Basis]
    geometry = energy.AffineResponse(np.eye(3))
    empty = factory(geometry, np.empty((0,3)))
    assert (empty.num_inputs, empty.num_outputs) == (12,0)
    assert empty.eval_value(np.zeros(12)).size == 0
    for points in [np.zeros((2,2)), np.full((1,3), np.nan)]:
        with pytest.raises(ValueError): factory(geometry, points)
    for A in [np.zeros((3,3)), np.diag([-1.,1.,1.]), np.zeros((2,3))]:
        with pytest.raises(ValueError): factory(energy.AffineResponse(A), np.zeros((1,3)))
