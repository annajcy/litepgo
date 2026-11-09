from batch_fixtures import BatchResponseFixture
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
    r = BatchResponseFixture(dof.ConstantDofMap(t, 2), dof.VertexDofMap(t, 1), local)
    del t, local
    gc.collect()
    np.testing.assert_allclose(r.evaluate(np.zeros(2)).value, expected @ [-2.0, -3.0])
    assert not saved[-1][2]
    np.testing.assert_array_equal(r.evaluate(np.zeros(2)).jacobian().toarray(), expected)
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
    r = BatchResponseFixture(dof.VertexDofMap(t, 2), dof.ElementDofMap(t, 1), local)
    r_evaluation = r.evaluate(np.arange(1.0, 11.0))
    J = r_evaluation.jacobian()
    value = r_evaluation.value
    np.testing.assert_array_equal(value, [16., 36.])
    assert J[0, 3] == 8 and J[1, 5] == 12 and J.nnz == 16
    J = r.evaluate(np.zeros(10)).jacobian()
    assert J.nnz == 16 and not J.data.any()


def test_bad_local_result_and_exception_propagation():
    t = topology()
    i, o = dof.ConstantDofMap(t, 2), dof.VertexDofMap(t, 1)
    bad = BatchResponseFixture(i, o, lambda ids, x, need: np.zeros((len(ids), 3)))
    with pytest.raises(RuntimeError, match="invalid batch"):
        bad.evaluate(np.zeros(2)).value
    bad = BatchResponseFixture(i, o, lambda ids, x, need: (np.zeros((len(ids), 4)), np.zeros((2, 4))))
    with pytest.raises(RuntimeError, match="invalid batch"):
        bad.evaluate(np.zeros(2)).jacobian()
    def fail(e, x, need):
        raise KeyError("element bug")
    with pytest.raises(KeyError, match="element bug"):
        BatchResponseFixture(i, o, fail).evaluate(np.zeros(2)).value
    with pytest.raises(ValueError, match="infeasible"):
        BatchResponseFixture(i, o, lambda *args: None).evaluate(np.zeros(2)).value


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
    r = BatchResponseFixture(i, o, local)
    expected = np.zeros((o.num_dofs, i.num_dofs))
    for e in range(2):
        expected[np.ix_(o.dof_indices(e), i.dof_indices(e))] += 1
    x = np.arange(i.num_dofs, dtype=float)
    np.testing.assert_array_equal(r.evaluate(x).value, expected @ x)
    np.testing.assert_array_equal(r.evaluate(x).jacobian().toarray(), expected)
