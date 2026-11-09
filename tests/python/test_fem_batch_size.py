import numpy as np
import pytest
from litepgo import dof, fem, mesh, tbb_thread_limit
from test_fem import constant_fields, rest_fields


@pytest.mark.parametrize('batch_size', [1, 16, 64, 256])
def test_native_energy_batch_size(batch_size):
    nodes = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    topology = mesh.TetTopology(4, np.tile(np.arange(4, dtype=np.int32), (17, 1)))
    mapping = dof.VertexDofMap(topology, 3)
    args = (mapping, rest_fields(fem.InterpolatedResponse[fem.TetP1Basis], mapping, nodes.ravel()),
            constant_fields(np.eye(3), 17), constant_fields(np.array([2., 3.]), 17))
    cls = fem.VolumetricFemEnergy[fem.TetP1Basis, fem.TetCentroidQuadrature]
    x = np.arange(12) * .001
    with tbb_thread_limit(1):
        expected = cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), *args).eval_hessian(x).toarray()
    with tbb_thread_limit(4):
        actual = cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), *args, batch_size=batch_size).eval_hessian(x).toarray()
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='batch size'):
        cls(fem.PointwiseVolumetricAdapter[fem.NeoHookean](), *args, batch_size=0)

    triangles = mesh.TriangleTopology(3, np.array([[0, 1, 2]], dtype=np.int32))
    stencils = mesh.make_triangle_mid_edge_stencils(triangles)
    mapping = dof.StencilDofMap(3, stencils.vertices, 3)
    rest = fem.TriangleMidEdgeRestData(nodes[:3], stencils.opposite[0])
    cls = fem.ShellFemEnergy
    args = (mapping, [rest], [.1], [np.array([2., 3.])])
    with tbb_thread_limit(1):
        expected = cls(fem.PointwiseShellAdapter[fem.KoiterStVK](), *args).eval_hessian(x[:9]).toarray()
    with tbb_thread_limit(4):
        actual = cls(fem.PointwiseShellAdapter[fem.KoiterStVK](), *args, batch_size=batch_size).eval_hessian(x[:9]).toarray()
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='batch size'):
        cls(fem.PointwiseShellAdapter[fem.KoiterStVK](), *args, batch_size=0)
