"""Coefficient sampling composes with constitutive force responses without Field glue."""
import numpy as np
import pytest
from litepgo import dof, energy, fem, mesh

@pytest.mark.parametrize("plastic", [False, True])
def test_sampled_parameter_force_chain_rule(plastic):
    B, Q = fem.TetP1Basis, fem.TetCentroidQuadrature
    nodes = np.vstack([B.coefficient_info(i).point for i in range(4)])
    geometry = [fem.InterpolatedResponse[B](nodes)]
    topology = mesh.TetTopology(4, np.array([[0, 1, 2, 3]], dtype=np.int32))
    displacement_map = dof.VertexDofMap(topology, 3)
    m = 9 if plastic else 2
    parameters = dof.ConstantDofMap(topology, m)
    fixed = [energy.AffineResponse(np.zeros((2 if plastic else 9, 3)),
             np.array([2., 3.]) if plastic else np.eye(3).ravel(order="F"))]
    cls = fem.VolumetricFemForceFpResponse if plastic else fem.VolumetricFemForceMaterialResponse
    force = cls[B, Q](fem.PointwiseVolumetricAdapter[fem.NeoHookean](), parameters, displacement_map, geometry, .1 * nodes.ravel(), fixed)
    sampling = fem.value_sampling_response[B](Q.points(), components=m)
    composed = energy.ComposedResponse(force, sampling)
    base = np.eye(3).ravel(order="F") if plastic else np.array([2., 3.])
    coefficients = np.tile(base, (4, 1))
    coefficients *= np.arange(4)[:, None] * .01 + 1
    c = coefficients.ravel()
    composed_evaluation = composed.evaluate(c)
    J = composed_evaluation.jacobian()
    value = composed_evaluation.value
    np.testing.assert_allclose(value, force.evaluate(coefficients.mean(axis=0)).value)
    h = 1e-6
    fd = np.column_stack([(composed.evaluate(c + h * d).value - composed.evaluate(c - h * d).value) / (2 * h) for d in np.eye(c.size)])
    np.testing.assert_allclose(J.toarray(), fd, atol=2e-9)
    assert J.shape == (12, 4*m)
