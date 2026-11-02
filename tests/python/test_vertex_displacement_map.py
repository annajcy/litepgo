import numpy as np
import pytest
from litepgo import mesh, dof


def test_exact_extracted_map_and_hermite_value_slots():
    X = np.array([[0.,0,0],[.25,.25,.25],[1,0,0],[0,1,0],[0,0,1]])
    cells = np.array([[1,2,3,4],[0,1,3,4],[0,2,1,4],[0,2,3,1]],dtype=np.int32)
    surface = mesh.extract_surface_with_vertices(mesh.TetMesh(X,cells))
    np.testing.assert_array_equal(surface.vertices,[0,2,3,4])
    np.testing.assert_array_equal(surface.mesh.positions,X[surface.vertices])
    for width in [3,24]:
        W = dof.build_vertex_displacement_map(surface.vertices,len(X)*width,width)
        assert W.nnz == 3*len(surface.vertices)
        np.testing.assert_array_equal(W.data,1)
        u = np.arange(len(X)*width,dtype=float)
        np.testing.assert_array_equal((W@u).reshape(-1,3),u.reshape(-1,width)[surface.vertices,:3])
        k = dof.SurfaceKinematics(surface.mesh.positions,W)
        np.testing.assert_array_equal(k.positions(u),surface.mesh.positions.ravel()+W@u)
    for ids in [[-1],[5]]:
        with pytest.raises(IndexError):
            dof.build_vertex_displacement_map(ids,15)
    with pytest.raises(ValueError):
        dof.build_vertex_displacement_map([],0,2)
    assert dof.build_vertex_displacement_map([],15).shape == (0,15)
