import numpy as np
import pytest
import scipy.sparse as sp

from litepgo import dof


def test_dof_reduction_splits_dofs():
    reduction = dof.DofReduction(6, np.array([4, 1]))
    assert (reduction.num_dofs, reduction.num_free) == (6, 4)
    np.testing.assert_array_equal(reduction.fixed_dofs, [1, 4])
    np.testing.assert_array_equal(reduction.free_dofs, [0, 2, 3, 5])
    assert [reduction.free_index(dof) for dof in range(6)] == [0, -1, 1, 2, -1, 3]
    with pytest.raises(IndexError):
        reduction.free_index(6)


@pytest.mark.parametrize("fixed", [[1, 1], [6], [-1]], ids=["duplicate", "too_large", "negative"])
def test_dof_reduction_rejects_invalid_fixed_dofs(fixed):
    with pytest.raises(ValueError):
        dof.DofReduction(6, fixed)


def test_dof_reduction_vectors_match_numpy_indexing():
    reduction = dof.DofReduction(6, [4, 1])
    full = np.linspace(10.0, 15.0, 6)
    np.testing.assert_array_equal(reduction.reduce(full), full[reduction.free_dofs])

    target = np.full(6, -7.0)
    reduction.expand(np.array([1.0, 2.0, 3.0, 4.0]), target)
    expected = np.full(6, -7.0)
    expected[reduction.free_dofs] = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize("convert", [sp.csc_matrix, sp.csr_matrix, sp.csc_array, lambda A: A.toarray()],
                         ids=["csc", "csr", "csc_array", "dense"])
def test_dof_reduction_matrix_matches_numpy_indexing(convert):
    reduction = dof.DofReduction(6, [4, 1])
    A = sp.random(6, 6, density=0.6, format="csc", random_state=0)
    A = (A + A.T).tocsc()
    reduced = reduction.reduce(convert(A))
    assert isinstance(reduced, sp.csc_matrix)
    free = reduction.free_dofs
    np.testing.assert_array_equal(reduced.toarray(), A.toarray()[np.ix_(free, free)])


def test_dof_reduction_with_everything_fixed():
    reduction = dof.DofReduction(3, [0, 1, 2])
    assert reduction.reduce(np.ones(3)).shape == (0,)
    assert reduction.reduce(sp.eye(3, format="csc")).shape == (0, 0)


def test_dof_reduction_rejects_lists_and_unwritable_targets():
    reduction = dof.DofReduction(4, [0])
    with pytest.raises(TypeError):
        reduction.reduce([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(TypeError):
        reduction.expand(np.ones(3), np.zeros(8)[::2])
    with pytest.raises(ValueError):
        reduction.expand(np.ones(2), np.zeros(4))


def test_dof_maps_mirror_cpp():
    from litepgo import mesh

    topology = mesh.TriangleTopology(4, np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32))
    x = np.arange(8.0)
    vertex = dof.VertexDofMap(topology, 2)
    assert (vertex.num_elements, vertex.num_dofs, vertex.dofs_per_vertex) == (2, 8, 2)
    np.testing.assert_array_equal(vertex.dof_indices(1), [4, 5, 2, 3, 6, 7])
    np.testing.assert_array_equal(vertex.gather(1, x), [[4, 5], [2, 3], [6, 7]])
    element = dof.ElementDofMap(topology, 3)
    assert (element.num_dofs, element.dofs_per_element) == (6, 3)
    np.testing.assert_array_equal(element.dof_indices(1), [3, 4, 5])
    constant = dof.ConstantDofMap(topology, 2)
    np.testing.assert_array_equal(constant.dof_indices(1), [0, 1])
    stencil = dof.StencilDofMap(4, [[0, 1, 2, 3], [2, 1, 3]], 2)
    np.testing.assert_array_equal(stencil.vertex_indices(0), [0, 1, 2, 3])
    np.testing.assert_array_equal(stencil.gather(1, x), [[4, 5], [2, 3], [6, 7]])
    with pytest.raises(IndexError):
        vertex.dof_indices(2)
    with pytest.raises(ValueError):
        vertex.gather(0, x[:7])
    with pytest.raises(ValueError):
        dof.StencilDofMap(4, [[0, 0, 1]], 2)


def test_surface_kinematics_general_map():
    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    W = sp.random(9, 5, density=0.5, format="csc", random_state=0)
    general = dof.SurfaceKinematics(rest, W)
    assert isinstance(general.displacement_map, sp.csc_matrix)
    np.testing.assert_array_equal(general.displacement_map.toarray(), W.toarray())
    np.testing.assert_allclose(general.displacements(np.ones(5)), W @ np.ones(5))
    with pytest.raises(ValueError):
        dof.SurfaceKinematics(rest, sp.eye(6))
