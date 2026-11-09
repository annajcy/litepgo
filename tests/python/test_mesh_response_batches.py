from batch_fixtures import BatchResponseFixture
import gc
import numpy as np
import pytest
from litepgo import dof, mesh, energy


@pytest.mark.parametrize("input_kind", ["constant", "element", "vertex", "stencil"])
@pytest.mark.parametrize("output_kind", ["constant", "element", "vertex", "stencil"])
@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_maps_and_batch_sizes(input_kind, output_kind, batch_size):
    t = mesh.TetTopology(6, np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]], dtype=np.int32))
    def make(kind):
        return {"constant": lambda: dof.ConstantDofMap(t, 2),
                "element": lambda: dof.ElementDofMap(t, 2),
                "vertex": lambda: dof.VertexDofMap(t, 2),
                "stencil": lambda: dof.StencilDofMap(6, [[0, 1], [1, 2, 3], [3, 4]], 2)}[kind]()
    i, o = make(input_kind), make(output_kind)
    rows = [o.dof_indices(e) for e in range(3)]
    cols = [i.dof_indices(e) for e in range(3)]
    expected = np.zeros((o.num_dofs, i.num_dofs))
    seen, saved = [], []
    for e in range(3):
        expected[np.ix_(rows[e], cols[e])] += e + 1
    def batch(ids, x, need):
        assert 0 < len(ids) <= batch_size
        assert len({(len(rows[e]), len(cols[e])) for e in ids}) == 1
        for b, e in enumerate(ids):
            np.testing.assert_array_equal(x[b], global_x[cols[e]])
        seen.extend(ids)
        saved.append((x, x.copy()))
        weights = np.asarray(ids) + 1
        no = len(rows[ids[0]])
        values = np.repeat((weights * x.sum(axis=1))[:, None], no, axis=1)
        J = np.repeat(weights, no)[:, None] * np.ones((len(ids)*no, x.shape[1]))
        return (values, J) if need else values
    obj = BatchResponseFixture(i, o, batch, batch_size)
    del t, i, o
    gc.collect()
    global_x = np.arange(obj.num_inputs, dtype=float)
    obj_evaluation = obj.evaluate(global_x)
    J = obj_evaluation.jacobian()
    value = obj_evaluation.value
    assert sorted(seen) == [0, 1, 2]
    np.testing.assert_array_equal(value, expected @ global_x)
    np.testing.assert_array_equal(J.toarray(), expected)
    global_x = -global_x
    np.testing.assert_array_equal(obj.evaluate(global_x).value, expected @ global_x)
    for x, copy in saved:
        np.testing.assert_array_equal(x, copy)


def test_zeroing_empty_errors_and_recovery():
    t = mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))
    i, o = dof.ConstantDofMap(t, 2), dof.VertexDofMap(t, 1)
    mode = "good"
    def batch(ids, x, need):
        if mode == "none":
            return None
        if mode == "raise":
            raise KeyError("callback error")
        y = np.full((len(ids), 4), x[0].sum()**2)
        J = np.full((len(ids)*4, 2), 2*x[0].sum())
        if mode == "bad_y": y = y[:, :3]
        if mode == "bad_j": J = J[:1]
        if mode == "nan": y[0, 0] = np.nan
        return (y, J) if need else y
    obj = BatchResponseFixture(i, o, batch, 1)
    nonzero = obj.evaluate(np.ones(2)).jacobian()
    zero = obj.evaluate(np.zeros(2)).jacobian()
    assert nonzero.nnz == zero.nnz == 10
    assert not zero.data.any()
    for mode in ("bad_y", "bad_j", "nan"):
        with pytest.raises(RuntimeError, match="invalid batch outputs"):
            obj.evaluate(np.ones(2)).jacobian()
    mode = "none"
    with pytest.raises(ValueError, match="infeasible"):
        obj.evaluate(np.ones(2)).value
    mode = "raise"
    with pytest.raises(KeyError, match="callback error"):
        obj.evaluate(np.ones(2)).value
    mode = "good"
    np.testing.assert_array_equal(obj.evaluate(np.ones(2)).jacobian().toarray(), nonzero.toarray())
    for size in (0, -1):
        with pytest.raises(ValueError, match="batch size"):
            BatchResponseFixture(i, o, batch, size)
    empty = dof.StencilDofMap(0, [], 1)
    def unreachable(*args):
        raise AssertionError("empty mesh callback")
    obj = BatchResponseFixture(empty, empty, unreachable)
    assert obj.evaluate(np.empty(0)).value.size == 0
    assert obj.evaluate(np.empty(0)).jacobian().shape == (0, 0)


@pytest.mark.parametrize("batch_size", [1, 32])
def test_grouping_preserves_original_element_sum_order(batch_size):
    topology = mesh.TetTopology(4, np.tile(np.arange(4, dtype=np.int32), (3, 1)))
    inputs = dof.StencilDofMap(2, [[0], [0, 1], [0]], 1)
    outputs = dof.ConstantDofMap(topology, 1)
    weights = np.array([1e16, -1e16, 1.])

    def batch(ids, x, need):
        values = (weights[ids] * x[:, 0])[:, None]
        J = np.zeros((len(ids), x.shape[1]))
        J[:, 0] = weights[ids]
        return (values, J) if need else values

    obj = BatchResponseFixture(inputs, outputs, batch, batch_size)
    obj_evaluation = obj.evaluate(np.ones(2))
    J = obj_evaluation.jacobian()
    value = obj_evaluation.value
    # Grouping evaluates elements 0,2 before 1; assembly must still sum 0,1,2.
    np.testing.assert_array_equal(value, [1.])
    np.testing.assert_array_equal(J.toarray(), [[1., 0.]])
    np.testing.assert_array_equal(obj.evaluate(np.ones(2)).jvp(np.ones(2)), [1.0])
    np.testing.assert_array_equal(obj.evaluate(np.ones(2)).vjp(np.ones(1)), [1.0, 0.0])
    np.testing.assert_array_equal(obj.evaluate(np.ones(2)).jacobian().toarray(), J.toarray())
