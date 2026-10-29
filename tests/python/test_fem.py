import gc
from pathlib import Path

import numpy as np
import pytest

from litepgo import dof, energy, fem, mesh

ASSETS = Path(__file__).resolve().parents[2] / "assets"

NEO_HOOKEAN_THETA = np.array([2.0, 3.0])


def test_templates_mirror_cpp_arguments():
    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.HexQ1Basis, fem.GaussLegendre3D[2]]
    assert Energy.__name__ == "VolumetricFemEnergy[NeoHookean, HexQ1Basis, GaussLegendre3D[2]]"
    assert issubclass(Energy, energy.Energy) and issubclass(Energy, fem.VolumetricFemEnergy)
    assert fem.InterpolatedField3D[fem.TetP1Basis] is fem.InterpolatedField3D[(fem.TetP1Basis,)]
    assert issubclass(fem.InterpolatedField3D[fem.TetP1Basis], fem.Field3D)
    assert fem.GaussLegendre3D[4].num_points == 64
    with pytest.raises(TypeError, match=r"available: .*GaussLegendre3D\[2\]"):
        fem.GaussLegendre3D[3]
    with pytest.raises(TypeError, match="VolumetricFemEnergy"):
        fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.GaussLegendre3D[2]]
    with pytest.raises(TypeError):
        fem.TetP1Basis()


@pytest.mark.parametrize(
    "basis,dim", [(fem.TriangleP1Basis, 2), (fem.QuadHermiteBasis, 2), (fem.TetP1Basis, 3), (fem.HexQ1Basis, 3)]
)
def test_lagrange_like_bases_partition_unity(basis, dim):
    rng = np.random.default_rng(0)
    xi = 0.3 * rng.random(dim)
    N = basis.eval_N(xi)
    assert N.shape == (basis.num_basis_functions,)
    if basis is not fem.QuadHermiteBasis:  # Hermite value functions alone sum to one
        assert N.sum() == pytest.approx(1.0)
    h = 1e-6
    for k in range(dim):
        step = np.zeros(dim)
        step[k] = h
        fd = (basis.eval_N(xi + step) - basis.eval_N(xi - step)) / (2 * h)
        np.testing.assert_allclose(basis.eval_dN_dxi(xi)[:, k], fd, atol=1e-8)


def test_basis_reference_nodes_interpolate():
    for i in range(fem.TetP1Basis.num_basis_functions):
        np.testing.assert_allclose(fem.TetP1Basis.eval_N(fem.TetP1Basis.reference_node(i)), np.eye(4)[i], atol=1e-15)


@pytest.mark.parametrize(
    "quadrature,volume",
    [
        (fem.TetCentroidQuadrature, 1 / 6),
        (fem.TriangleCentroidQuadrature, 1 / 2),
        (fem.GaussLegendre2D[2], 1.0),
        (fem.GaussLegendre3D[4], 1.0),
    ],
)
def test_quadrature_weights(quadrature, volume):
    assert quadrature.points().shape[0] == quadrature.num_points
    assert quadrature.weights().sum() == pytest.approx(volume)


def test_fields():
    coeffs = np.array([fem.HexQ1Basis.reference_node(v) for v in range(8)]) * [2.0, 1.0, 3.0]
    X = fem.InterpolatedField3D[fem.HexQ1Basis](coeffs)
    xi = np.array([0.2, 0.5, 0.9])
    np.testing.assert_allclose(X.eval_value(xi), [0.4, 0.5, 2.7])
    np.testing.assert_allclose(X.eval_dvalue_dxi(xi), np.diag([2.0, 1.0, 3.0]), atol=1e-15)
    assert (X.num_components, X.num_dofs) == (3, 24)
    constant = fem.IdentityField3D(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(constant.eval_value(xi), [1.0, 2.0])
    with pytest.raises(ValueError):
        fem.InterpolatedField3D[fem.HexQ1Basis](coeffs[:7])


def test_deformation_gradient_keeps_its_fields_alive():
    def make():
        nodes = np.array([fem.TetP1Basis.reference_node(v) for v in range(4)])
        u = fem.InterpolatedField3D[fem.TetP1Basis](0.1 * nodes)
        X = fem.InterpolatedField3D[fem.TetP1Basis](nodes)
        return fem.DeformationGradientField3D(u, X)

    F = make()
    gc.collect()
    np.testing.assert_allclose(F.eval_value(np.full(3, 0.25)), (1.1 * np.eye(3)).ravel(order="F"))


def test_materials_pointwise():
    I = np.eye(3)
    assert fem.NeoHookean.eval_W(I, I, NEO_HOOKEAN_THETA) == pytest.approx(0.0, abs=1e-15)
    F = I + 0.1 * np.arange(9.0).reshape(3, 3) / 9
    P = fem.NeoHookean.eval_dW_dF(F, I, NEO_HOOKEAN_THETA)
    h = 1e-6
    for i in range(3):
        for j in range(3):
            dF = np.zeros((3, 3))
            dF[i, j] = h
            fd = (fem.NeoHookean.eval_W(F + dF, I, NEO_HOOKEAN_THETA) - fem.NeoHookean.eval_W(F - dF, I, NEO_HOOKEAN_THETA)) / (2 * h)
            assert P[i, j] == pytest.approx(fd, abs=1e-8)
    assert fem.NeoHookean.eval_feasible_step(I, I, NEO_HOOKEAN_THETA, -2 * I, 0 * I, 0 * NEO_HOOKEAN_THETA) < 0.5
    a = np.array([[1.0, 0.2], [0.2, 1.5]])
    assert fem.KoiterStVK.eval_W(a, 0 * a, a, 0 * a, 0.1, np.array([1.0, 2.0])) == 0.0
    with pytest.raises(ValueError):
        fem.NeoHookean.eval_W(I, I, np.array([1.0, -1.0]))


def rest_fields(XField, rest_map, positions):
    return [XField(rest_map.gather(e, positions)) for e in range(rest_map.num_elements)]


def constant_fields(value, count):
    return [fem.IdentityField3D(np.asarray(value, dtype=float).ravel())] * count


def volumetric_energy(data, basis, quadrature, dofs_per_vertex, theta_fields=None):
    rest_map = dof.VertexDofMap(data.topology, 3)
    X_fields = rest_fields(fem.InterpolatedField3D[type_of_rest_basis(data)], rest_map, data.positions.ravel())
    n = data.topology.num_cells
    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, basis, quadrature]
    return Energy(
        dof.VertexDofMap(data.topology, dofs_per_vertex),
        X_fields,
        constant_fields(np.eye(3), n),
        theta_fields or constant_fields(NEO_HOOKEAN_THETA, n),
    )


def type_of_rest_basis(data):
    return fem.TetP1Basis if isinstance(data, mesh.TetMesh) else fem.HexQ1Basis


def test_single_tet_energy_matches_material_times_volume():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    data = mesh.TetMesh(nodes, np.array([[0, 1, 2, 3]], dtype=np.int32))
    elastic = volumetric_energy(data, fem.TetP1Basis, fem.TetCentroidQuadrature, 3)
    A = np.eye(3) + 0.1 * np.array([[1, 2, 0], [0, 1, 3], [1, 0, 2.0]])
    u = (nodes @ (A - np.eye(3)).T).ravel()
    expected = fem.NeoHookean.eval_W(A, np.eye(3), NEO_HOOKEAN_THETA) / 6
    assert elastic.eval_value(u) == pytest.approx(expected, rel=1e-13)
    assert isinstance(elastic, fem.VolumetricFemEnergy) and isinstance(elastic, energy.Energy)


def test_erased_field_types_can_be_mixed():
    data = mesh.read_veg(ASSETS / "box-hex.veg")
    n = data.topology.num_cells
    Q1 = fem.InterpolatedField3D[fem.HexQ1Basis]
    # theta interpolated from Q1 nodal values equals the constant field everywhere.
    interpolated_theta = [Q1(np.tile(NEO_HOOKEAN_THETA, (8, 1)))] * n
    constant = volumetric_energy(data, fem.HexHermiteBasis, fem.GaussLegendre3D[2], 24)
    mixed = volumetric_energy(data, fem.HexHermiteBasis, fem.GaussLegendre3D[2], 24, interpolated_theta)
    u = 1e-3 * np.sin(np.arange(constant.num_dofs))
    assert mixed.eval_value(u) == pytest.approx(constant.eval_value(u), rel=1e-14)
    np.testing.assert_allclose(mixed.eval_gradient(u), constant.eval_gradient(u), rtol=1e-12, atol=1e-14)


def test_energy_rejects_bad_fields():
    data = mesh.read_veg(ASSETS / "box-tet.veg")
    Energy = fem.VolumetricFemEnergy[fem.NeoHookean, fem.TetP1Basis, fem.TetCentroidQuadrature]
    rest_map = dof.VertexDofMap(data.topology, 3)
    X = rest_fields(fem.InterpolatedField3D[fem.TetP1Basis], rest_map, data.positions.ravel())
    n = data.topology.num_cells
    with pytest.raises(ValueError):
        Energy(rest_map, X[:-1], constant_fields(np.eye(3), n), constant_fields(NEO_HOOKEAN_THETA, n))
    with pytest.raises(ValueError):
        Energy(rest_map, X, constant_fields(np.eye(2), n), constant_fields(NEO_HOOKEAN_THETA, n))
    with pytest.raises(TypeError):
        Energy(rest_map, X, [np.eye(3)] * n, constant_fields(NEO_HOOKEAN_THETA, n))
