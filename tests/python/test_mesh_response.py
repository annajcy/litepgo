import gc
import numpy as np
import pytest

from litepgo import dof, mesh, energy, solver


def topology():
    return mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))


def test_shared_scatter_and_solver():
    t = topology()
    A = np.arange(1., 9.).reshape(4, 2)
    expected = np.array([[1., 2.], [5., 8.], [11., 14.], [17., 20.], [14., 16.]])
    saved = []
    def local(ids, x, need_jacobian):
        saved.append((x, x.copy(), need_jacobian))
        B = (np.asarray(ids) + 1)[:, None, None] * A
        y = np.einsum("bij,bj->bi", B, x - [2., 3.])
        return (y, B.reshape(-1, 2)) if need_jacobian else y
    r = energy.CustomMeshResponse(dof.ConstantDofMap(t, 2), dof.VertexDofMap(t, 1), local)
    del t, local
    gc.collect()
    assert r.num_inputs == 2 and r.num_outputs == 5 and r.has_fixed_pattern
    np.testing.assert_allclose(r.eval_value(np.zeros(2)), expected @ [-2., -3.])
    assert not saved[-1][2]
    np.testing.assert_array_equal(r.eval_jacobian(np.zeros(2)).toarray(), expected)
    result = solver.gauss_newton_solve(r, np.zeros(2), solver.GaussNewtonOptions(gradient_tolerance=1e-8))
    assert result.status == solver.GaussNewtonStatus.Converged
    np.testing.assert_allclose(result.x, [2., 3.], atol=1e-7)
    for x, copy, _ in saved:
        np.testing.assert_array_equal(x, copy)


def test_matrix_gather_and_zero_values_preserve_pattern():
    t = topology()
    def local(ids, x, need):
        assert x.shape == (len(ids), 8)
        y = x[:, 3:4]**2
        J = np.zeros((len(ids), 8))
        J[:, 3] = 2*x[:, 3]
        return (y, J) if need else y
    r = energy.CustomMeshResponse(dof.VertexDofMap(t, 2), dof.ElementDofMap(t, 1), local)
    value, J = r.eval(np.arange(1., 11.), need_jacobian=True)
    np.testing.assert_array_equal(value, [16., 36.])
    assert J[0, 3] == 8 and J[1, 5] == 12 and J.nnz == 16
    J = r.eval_jacobian(np.zeros(10))
    assert J.nnz == 16 and not J.data.any()


def test_bad_local_result_and_exception_propagation():
    t = topology()
    i, o = dof.ConstantDofMap(t, 2), dof.VertexDofMap(t, 1)
    bad = energy.CustomMeshResponse(i, o, lambda ids, x, need: np.zeros((len(ids), 3)))
    with pytest.raises(RuntimeError, match="invalid batch"):
        bad.eval_value(np.zeros(2))
    bad = energy.CustomMeshResponse(i, o, lambda ids, x, need: (np.zeros((len(ids), 4)), np.zeros((2, 4))))
    with pytest.raises(RuntimeError, match="invalid batch"):
        bad.eval_jacobian(np.zeros(2))
    def fail(e, x, need):
        raise KeyError("element bug")
    with pytest.raises(KeyError, match="element bug"):
        energy.CustomMeshResponse(i, o, fail).eval_value(np.zeros(2))
    with pytest.raises(ValueError, match="infeasible"):
        energy.CustomMeshResponse(i, o, lambda *args: None).eval_value(np.zeros(2))


@pytest.mark.parametrize("input_kind", ["constant", "element", "vertex", "stencil"])
@pytest.mark.parametrize("output_kind", ["constant", "element", "vertex", "stencil"])
def test_bound_map_combinations(input_kind, output_kind):
    t = topology()
    def make(kind):
        if kind == "constant":
            return dof.ConstantDofMap(t, 2)
        if kind == "element":
            return dof.ElementDofMap(t, 2)
        if kind == "vertex":
            return dof.VertexDofMap(t, 2)
        return dof.StencilDofMap(5, [[0, 2], [2, 3, 4]], 2)
    i, o = make(input_kind), make(output_kind)
    def local(ids, x, need):
        count = len(o.dof_indices(ids[0]))
        value = np.repeat(x.sum(axis=1)[:, None], count, axis=1)
        return (value, np.ones((len(ids)*count, x.shape[1]))) if need else value
    r = energy.CustomMeshResponse(i, o, local)
    expected = np.zeros((o.num_dofs, i.num_dofs))
    for e in range(2):
        expected[np.ix_(o.dof_indices(e), i.dof_indices(e))] += 1
    x = np.arange(i.num_dofs, dtype=float)
    np.testing.assert_array_equal(r.eval_value(x), expected @ x)
    np.testing.assert_array_equal(r.eval_jacobian(x).toarray(), expected)
