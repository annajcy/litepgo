import gc
import numpy as np
import pytest
from litepgo import dof, fem, mesh


@pytest.mark.parametrize("basis,quadrature,topology_type", [
    (fem.TetP1Basis, fem.TetCentroidQuadrature, mesh.TetTopology),
    (fem.HexQ1Basis, fem.GaussLegendre3D[2], mesh.HexTopology),
])
@pytest.mark.parametrize("nodal_parameters", [False, True])
def test_material_local_response_matches_energy_and_parameter_linearity(basis, quadrature, topology_type, nodal_parameters):
    n = 4 if basis is fem.TetP1Basis else 8
    nodes = 2 * np.array([basis.coefficient_info(i).point for i in range(n)])
    Field = fem.InterpolatedField3D[basis]
    X = Field(nodes)
    F_value = np.array([[1.2, .3, -.1], [.1, .9, .2], [-.05, .1, 1.1]])
    Fp_value = np.array([[1.3, .2, .1], [0., .9, -.1], [.1, .2, 1.2]])
    displacement = nodes @ (F_value - np.eye(3)).T
    F = fem.DeformationGradientField3D(Field(displacement), X)
    Fp = fem.IdentityField3D(Fp_value.ravel(order="F"))
    coefficients = np.column_stack([2 + .1*nodes[:, 0], 3 + .2*nodes[:, 1]]) if nodal_parameters else np.array([2., 3.])
    theta = Field(coefficients) if nodal_parameters else fem.IdentityField3D(coefficients)
    Local = fem.VolumetricMaterialForceLocalResponse3D[fem.NeoHookean]
    local = Local(F, Fp, theta)
    energy = fem.VolumetricMaterialLocalEnergyF3D[fem.NeoHookean](F, Fp, theta)
    xi = np.array([.2, .1, .3])
    assert isinstance(local, fem.LocalResponse3D)
    assert isinstance(local, fem.VolumetricMaterialForceLocalResponse3D)
    assert local.num_inputs == coefficients.size and local.num_outputs == 3*n and local.has_jacobian
    value, J = local.eval_value(xi), local.eval_jacobian(xi)
    assert J.shape == (3*n, coefficients.size)
    np.testing.assert_allclose(value, energy.eval_gradient(xi), rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(J @ coefficients.ravel(), value, rtol=1e-12, atol=1e-13)
    weighted, weighted_J = local.accumulate(xi, -.25, jacobian=True)
    np.testing.assert_allclose(weighted, -.25*value, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(weighted_J, -.25*J, rtol=1e-12, atol=1e-13)
    assert local.accumulate(xi, 1)[1] is None
    element = fem.VolumetricElement3D[quadrature](X)
    force, B = element.eval(local, jacobian=True)
    assert element.eval(local)[1] is None
    _, g, _ = element.eval(energy, gradient=True)
    np.testing.assert_allclose(force, g, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(B @ coefficients.ravel(), force, rtol=1e-12, atol=1e-13)
    topology = topology_type(n, np.arange(n, dtype=np.int32)[None])
    whole = fem.VolumetricFemEnergy[fem.NeoHookean, basis, quadrature](
        dof.VertexDofMap(topology, 3), [X], [Fp], [theta])
    np.testing.assert_allclose(force, whole.eval_gradient(displacement.ravel()), rtol=1e-12, atol=1e-13)
    if nodal_parameters:
        constant = Local(F, Fp, fem.IdentityField3D(np.array([2., 3.])))
        constant_B = element.eval(constant, jacobian=True)[1]
        np.testing.assert_allclose(B[:, 0::2].sum(axis=1), constant_B[:, 0], rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(B[:, 1::2].sum(axis=1), constant_B[:, 1], rtol=1e-12, atol=1e-13)


def test_material_local_response_owns_python_fields_and_rejects_invalid_inputs():
    Local = fem.VolumetricMaterialForceLocalResponse3D[fem.NeoHookean]
    def make():
        return Local(fem.IdentityField3D((1.2*np.eye(3)).ravel(order="F")),
                     fem.IdentityField3D(np.eye(3).ravel(order="F")),
                     fem.IdentityField3D(np.array([2., 3.])))
    local = make()
    gc.collect()
    xi = np.zeros(3)
    assert local.eval_jacobian(xi).shape == (9, 2)
    np.testing.assert_allclose(local.eval_value(xi), local.eval_jacobian(xi) @ [2., 3.])
    with pytest.raises(ValueError):
        local.accumulate(xi, np.inf)
    with pytest.raises(ValueError):
        local.eval_jacobian(np.array([np.nan, 0, 0]))
    with pytest.raises(ValueError):
        Local(fem.IdentityField3D(np.ones(8)), fem.IdentityField3D(np.ones(9)),
              fem.IdentityField3D(np.array([2., 3.])))
    invalid = Local(fem.IdentityField3D(np.eye(3).ravel()), fem.IdentityField3D(np.eye(3).ravel()),
                    fem.IdentityField3D(np.array([2., 0.])))
    with pytest.raises(ValueError):
        invalid.eval_value(xi)
    with pytest.raises(ValueError):
        invalid.eval_jacobian(xi)
