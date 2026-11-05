import numpy as np
import pytest
from litepgo import fem, mesh, dof, contact


def model():
    X = np.array([fem.HexQ1Basis.coefficient_info(i).point for i in range(8)]) * [2.,3.,4.]
    right = X + [2.,0,0]
    X = np.vstack([X,right[[1,2,5,6]]])
    cells = np.array([[0,1,2,3,4,5,6,7],[1,8,9,2,5,10,11,6]],dtype=np.int32)
    topology = mesh.HexTopology(len(X),cells)
    fields = [fem.InterpolatedField3D[fem.HexQ1Basis](X[c]) for c in cells]
    return X,topology,fields


@pytest.mark.parametrize('n',[1,2,3])
def test_conforming_two_hex_boundary(n):
    X,topology,fields = model()
    sampled = fem.sample_boundary[fem.HexQ1Basis](topology,fields,n)
    surface = sampled.mesh
    assert surface.topology.num_cells == 20*n*n
    assert surface.topology.num_vertices == (2*n+1)*(n+1)**2 - (2*n-1)*(n-1)**2
    if n == 1:
        extracted = mesh.extract_surface(mesh.HexMesh(X,topology))
        np.testing.assert_array_equal(surface.positions,extracted.positions)
        np.testing.assert_array_equal(surface.cells,extracted.cells)
    edges = {}
    volume = 0.
    for tri in surface.cells:
        a,b,c = surface.positions[tri]
        volume += np.dot(a,np.cross(b,c))/6
        for i in range(3):
            edge = (tri[i],tri[(i+1)%3])
            edges[edge] = edges.get(edge,0)+1
    assert volume == pytest.approx(48.)
    assert all(count == 1 and edges[(b,a)] == 1 for (a,b),count in edges.items())
    for p,e,xi in zip(surface.positions,sampled.embedding.element_ids,sampled.embedding.xi):
        np.testing.assert_allclose(fields[e].eval_value(xi),p,atol=1e-12)
    # The resulting topology is accepted as a contact surface.
    barrier = contact.IPCContactEnergy(surface,contact.IPCContactOptions(dhat=.01),contact.SpatialHash())
    assert barrier.num_vertices == len(surface.positions)


def test_hermite_derivatives_move_face_samples_but_not_corners():
    X,topology,fields = model()
    layout = dof.VertexDofMap(topology,24)
    u = np.zeros(layout.num_dofs)
    u[3+2] = 1. # vertex 0: z displacement derivative with respect to physical X
    for n in [1,4]:
        sampled = fem.sample_boundary[fem.HexQ1Basis](topology,fields,n)
        W = fem.build_point_displacement_map[fem.HexHermiteBasis](sampled.embedding,layout,fields)
        displacement = (W@u).reshape(-1,3)
        if n == 1:
            assert W.nnz == 3*len(sampled.mesh.positions)
            np.testing.assert_array_equal(displacement,0)
        else:
            assert np.max(np.abs(displacement)) > .01
            np.testing.assert_array_equal(displacement[:len(X)],0)
        for i,(e,xi) in enumerate(zip(sampled.embedding.element_ids,sampled.embedding.xi)):
            coeffs = fem.BasisDofTransform[fem.HexHermiteBasis](fields[e]).to_coefficients(layout.gather(e,u))
            expected = fem.InterpolatedField3D[fem.HexHermiteBasis](coeffs).eval_value(xi)
            np.testing.assert_allclose(displacement[i],expected,atol=1e-12)


def test_boundary_input_checks():
    X,topology,fields = model()
    for n in [0,-1]:
        with pytest.raises(ValueError):
            fem.sample_boundary[fem.HexQ1Basis](topology,fields,n)
    with pytest.raises(ValueError):
        fem.sample_boundary[fem.HexQ1Basis](topology,fields[:1],2)
    bad = fem.InterpolatedField3D[fem.HexQ1Basis](X[[1,8,9,2,5,10,11,6]] + [0,.01,0])
    with pytest.raises(ValueError,match='inconsistent'):
        fem.sample_boundary[fem.HexQ1Basis](topology,[fields[0],bad],2)
    cells = np.tile(np.arange(8,dtype=np.int32),(3,1))
    with pytest.raises(ValueError,match='more than two'):
        fem.sample_boundary[fem.HexQ1Basis](mesh.HexTopology(8,cells),[fields[0]]*3,2)
    empty = fem.sample_boundary[fem.HexQ1Basis](mesh.HexTopology(0,np.empty((0,8),dtype=np.int32)),[],2)
    assert empty.mesh.positions.shape == (0,3)
    assert empty.embedding.xi.shape == (0,3)
