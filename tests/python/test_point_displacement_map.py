import numpy as np
import pytest
import scipy.sparse as sp
from litepgo import fem, dof, mesh


@pytest.mark.parametrize('basis,width', [(fem.TetP1Basis, 3), (fem.HexQ1Basis, 3), (fem.HexHermiteBasis, 24)])
def test_map_matches_physical_dof_interpolation(basis, width):
    tet = basis is fem.TetP1Basis
    geometry = fem.TetP1Basis if tet else fem.HexQ1Basis
    count = 4 if tet else 8
    X = np.array([geometry.reference_node(i) for i in range(count)]) * [2., 3., 4.]
    cells = np.array([np.arange(count), np.arange(count) + count], dtype=np.int32)
    topology = (mesh.TetTopology if tet else mesh.HexTopology)(2 * count, cells)
    fields = [fem.InterpolatedField3D[geometry](X), fem.InterpolatedField3D[geometry](X + [5., 0, 0])]
    xi = np.array([[.1, .2, .3], [.2, .1, .2]])
    points = np.array([fields[1].eval_value(xi[0]), fields[0].eval_value(xi[1])])
    embedding = fem.ElementLocator[geometry](fields).locate(points)
    layout = dof.VertexDofMap(topology, width)
    W = fem.build_point_displacement_map[basis](embedding, layout, fields)
    assert sp.isspmatrix_csc(W)
    assert W.shape == (6, layout.num_dofs)
    u = np.random.default_rng(42).normal(size=layout.num_dofs)
    expected = []
    for e, local_xi in zip(embedding.element_ids, embedding.xi):
        transform = fem.BasisDofTransform[basis](fields[e])
        coefficients = transform.to_coefficients(layout.gather(e, u))
        expected.extend(fem.InterpolatedField3D[basis](coefficients).eval_value(local_xi))
    np.testing.assert_allclose(W @ u, expected, atol=1e-12)
    kinematics = dof.SurfaceKinematics(points, W)
    np.testing.assert_allclose(kinematics.positions(u), points.ravel() + expected)
    # An affine physical displacement must reproduce exactly, including Hermite derivative slots.
    F = np.array([[.2, -.1, .3], [.4, .5, -.2], [.1, .3, .6]])
    nodal = np.zeros((2 * count, width))
    world_nodes = np.vstack([X, X + [5., 0, 0]])
    nodal[:, :3] = world_nodes @ F.T + [1., 2., 3.]
    if width == 24:
        nodal[:, 3:6], nodal[:, 6:9], nodal[:, 9:12] = F[:, 0], F[:, 1], F[:, 2]
    np.testing.assert_allclose((W @ nodal.ravel()).reshape(-1, 3), points @ F.T + [1., 2., 3.], atol=1e-10)
    with pytest.raises(ValueError):
        fem.build_point_displacement_map[basis](embedding, layout, fields[:1])
    with pytest.raises(ValueError):
        fem.build_point_displacement_map[basis](embedding, dof.VertexDofMap(topology, width+1), fields)
    empty = fem.ElementLocator[geometry](fields).locate(np.empty((0, 3)))
    assert fem.build_point_displacement_map[basis](empty, layout, fields).shape == (0, layout.num_dofs)
