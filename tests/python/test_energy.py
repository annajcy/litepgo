from batch_fixtures import BatchEnergyFixture
import gc

import numpy as np
import pytest
import scipy.sparse as sp

from litepgo.energy import (
    AffineMappedEnergy,
    Energy,
    HessianKind,
    QuadraticEnergy,
    VertexPullingEnergy,
    WeightedEnergySum,
)


def laplacian(n):
    return sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csc")


def quadratic(n, seed):
    rng = np.random.default_rng(seed)
    B = sp.random(n, n, density=0.4, format="csc", random_state=seed)
    return QuadraticEnergy((B + B.T + 4.0 * sp.identity(n)).tocsc(), rng.standard_normal(n))


def test_quadratic_energy_matches_numpy():
    A = laplacian(5)
    b = np.linspace(1.0, 2.0, 5)
    energy = QuadraticEnergy(A, b)
    x = np.linspace(-1.0, 1.0, 5)
    dense = A.toarray()
    assert energy.eval_value(x) == pytest.approx(0.5 * x @ dense @ x + b @ x)
    np.testing.assert_allclose(energy.eval_gradient(x), dense @ x + b)
    hessian = energy.eval_hessian(x)
    assert isinstance(hessian, sp.csc_matrix)
    np.testing.assert_array_equal(hessian.toarray(), dense)
    np.testing.assert_array_equal(energy.eval_projected_hessian(x).toarray(), dense)
    assert energy.num_dofs == 5 and energy.has_fixed_pattern


def test_quadratic_energy_rejects_invalid_input():
    with pytest.raises(ValueError, match="symmetric"):
        QuadraticEnergy(sp.csc_matrix(np.array([[1.0, 2.0], [0.0, 1.0]])))
    with pytest.raises(ValueError):
        QuadraticEnergy(laplacian(3), np.ones(2))


def test_eval_returns_what_was_asked():
    energy = QuadraticEnergy(laplacian(3))
    x = np.array([1.0, 0.0, -1.0])
    value, gradient, hessian = energy.eval(x)
    assert gradient is None and hessian is None
    value, gradient, hessian = energy.eval(x, gradient=True, hessian=HessianKind.Projected)
    assert value == energy.eval_value(x)
    np.testing.assert_array_equal(gradient, energy.eval_gradient(x))
    np.testing.assert_array_equal(hessian.toarray(), laplacian(3).toarray())


def test_pattern_queries():
    energy = QuadraticEnergy(laplacian(4))
    pattern = energy.hessian_pattern
    assert isinstance(pattern, sp.csc_matrix)
    assert pattern.nnz == laplacian(4).nnz and not pattern.data.any()
    assert energy.update_pattern(np.zeros(4)) is False
    assert energy.max_feasible_step(np.zeros(4), np.ones(4)) == 1.0


def test_weighted_energy_sum_matches_the_weighted_terms():
    first, second = quadratic(6, 0), quadratic(6, 1)
    total = WeightedEnergySum([(first, 2.0), (second, 0.5)])
    x = np.linspace(-1.0, 1.0, 6)
    assert total.eval_value(x) == pytest.approx(2.0 * first.eval_value(x) + 0.5 * second.eval_value(x))
    np.testing.assert_allclose(total.eval_gradient(x), 2.0 * first.eval_gradient(x) + 0.5 * second.eval_gradient(x))
    np.testing.assert_allclose(total.eval_hessian(x).toarray(),
                               2.0 * first.eval_hessian(x).toarray() + 0.5 * second.eval_hessian(x).toarray())


@pytest.mark.parametrize("terms", [[], [(QuadraticEnergy(laplacian(3)), 0.0)],
                                   [(QuadraticEnergy(laplacian(3)), float("nan"))],
                                   [(QuadraticEnergy(laplacian(3)), 1.0), (QuadraticEnergy(laplacian(4)), 1.0)]],
                         ids=["empty", "zero_weight", "nan_weight", "mismatched_dofs"])
def test_weighted_energy_sum_rejects_invalid_terms(terms):
    with pytest.raises(ValueError):
        WeightedEnergySum(terms)


def test_affine_mapped_energy_matches_dense_products():
    inner = quadratic(5, 2)
    T = sp.csc_matrix(np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.0], [1.0, -1.0]]))
    offset = np.linspace(0.0, 1.0, 5)
    mapped = AffineMappedEnergy(inner, T, offset)
    z = np.array([0.3, -0.8])
    x = T @ z + offset
    dense_T = T.toarray()
    assert mapped.num_dofs == 2
    assert mapped.eval_value(z) == pytest.approx(inner.eval_value(x))
    np.testing.assert_allclose(mapped.eval_gradient(z), dense_T.T @ inner.eval_gradient(x))
    np.testing.assert_allclose(mapped.eval_hessian(z).toarray(), dense_T.T @ inner.eval_hessian(x).toarray() @ dense_T)


def test_weighted_energy_sum_allows_negative_weights():
    first, second = quadratic(4, 6), quadratic(4, 7)
    total = WeightedEnergySum([(first, 1.0), (second, -0.25)])
    x = np.linspace(-1.0, 1.0, 4)
    assert total.eval_value(x) == pytest.approx(first.eval_value(x) - 0.25 * second.eval_value(x))


def test_affine_mapped_energy_defaults_to_zero_offset_and_checks_shapes():
    inner = quadratic(4, 3)
    mapped = AffineMappedEnergy(inner, sp.identity(4, format="csc"))
    z = np.linspace(-1.0, 1.0, 4)
    assert mapped.eval_value(z) == pytest.approx(inner.eval_value(z))
    with pytest.raises(ValueError):
        AffineMappedEnergy(inner, sp.identity(3, format="csc"))


def test_composites_keep_their_children_alive():
    first, second = quadratic(4, 4), quadratic(4, 5)
    x = np.ones(4)
    expected = first.eval_value(x) + second.eval_value(x)
    total = WeightedEnergySum([(first, 1.0), (second, 1.0)])
    mapped = AffineMappedEnergy(total, sp.identity(4, format="csc"))
    del first, second, total
    gc.collect()
    assert mapped.eval_value(x) == pytest.approx(expected)


def test_checks_vector_sizes_and_types():
    energy = QuadraticEnergy(laplacian(3))
    with pytest.raises(ValueError):
        energy.eval_value(np.ones(4))
    with pytest.raises(TypeError):
        energy.eval_value([1.0, 2.0, 3.0])


def test_energy_base_is_abstract():
    with pytest.raises(TypeError):
        Energy()
    assert issubclass(WeightedEnergySum, Energy) and issubclass(AffineMappedEnergy, Energy)


def test_vertex_pulling_energy_matches_numpy():
    dofs = np.array([[0, 1, 2], [6, 4, 5]], dtype=np.int32)
    targets = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -1.0]])
    energy = VertexPullingEnergy(7, dofs, targets, 2.0)
    x = np.linspace(-1.0, 2.0, 7)
    gap = x[dofs] - targets
    assert energy.eval_value(x) == pytest.approx(gap.ravel() @ gap.ravel())
    expected_gradient = np.zeros(7)
    expected_gradient[dofs.ravel()] = 2.0 * gap.ravel()
    np.testing.assert_allclose(energy.eval_gradient(x), expected_gradient)
    expected_hessian = np.zeros((7, 7))
    expected_hessian[dofs.ravel(), dofs.ravel()] = 2.0
    np.testing.assert_array_equal(energy.eval_projected_hessian(x).toarray(), expected_hessian)
    with pytest.raises(ValueError, match="distinct"):
        VertexPullingEnergy(7, np.array([[0, 1, 2], [2, 4, 5]], dtype=np.int32), targets, 2.0)
    with pytest.raises(ValueError):
        VertexPullingEnergy(7, dofs, targets[:1], 2.0)


def test_mesh_energy_stencil_assembly_and_callback_lifetime():
    from litepgo.dof import StencilDofMap

    retained = []
    def local(ids, x, want_g, want_h):
        retained.append((x, x.copy()))
        d = x[:, :3] - x[:, 3:]
        H = np.kron([[1., -1.], [-1., 1.]], np.eye(3))
        return (.5*np.sum(d*d,axis=1), np.concatenate([d,-d],axis=1) if want_g else None,
                np.tile(H,(len(ids),1)) if want_h else None)

    mapping = StencilDofMap(3, [[0, 1], [1, 2]], 3)
    assembled = BatchEnergyFixture(mapping, local)
    del mapping, local
    gc.collect()
    x = np.arange(9, dtype=float)
    H = np.kron([[1., -1., 0.], [-1., 2., -1.], [0., -1., 1.]], np.eye(3))
    assert assembled.eval_value(x) == pytest.approx(0.5 * x @ H @ x)
    np.testing.assert_allclose(assembled.eval_gradient(x), H @ x)
    np.testing.assert_allclose(assembled.eval_hessian(x).toarray(), H)
    np.testing.assert_allclose(assembled.eval_projected_hessian(x).toarray(), H, atol=1e-14)
    for owned, snapshot in retained:
        np.testing.assert_array_equal(owned, snapshot)
    assert assembled.max_feasible_step(x, x) == 1.


def test_mesh_energy_callback_errors_and_step():
    from litepgo.dof import StencilDofMap

    mapping = StencilDofMap(1, [[0]], 3)
    x = np.ones(3)
    def fail(*args):
        raise ValueError('local callback failed')
    with pytest.raises(ValueError, match='local callback failed'):
        BatchEnergyFixture(mapping, fail).eval_value(x)
    with pytest.raises(ValueError, match='Missing batch gradients'):
        BatchEnergyFixture(mapping, lambda *args: (np.zeros(1), None, None)).eval_gradient(x)
    with pytest.raises(RuntimeError, match='invalid batch outputs'):
        BatchEnergyFixture(mapping, lambda *args: (np.zeros(1), np.zeros((1,2)), None)).eval_gradient(x)
    with pytest.raises(RuntimeError, match='invalid batch outputs'):
        BatchEnergyFixture(mapping, lambda *args: (np.array([np.nan]), None, None)).eval_value(x)
    bounded = BatchEnergyFixture(mapping, lambda *args: (np.zeros(1), None, None), step=lambda ids, x, dx: np.full(len(ids), .25))
    assert bounded.max_feasible_step(x, x) == 0.25


@pytest.mark.parametrize('batch_size', [1, 2, 256])
def test_batched_mesh_energy_variable_stencils(batch_size):
    from litepgo.dof import StencilDofMap
    mapping = StencilDofMap(4, [[0, 1], [1, 2, 3], [2, 3], [0]], 3)
    seen = []
    def batch(ids, x, g, h):
        seen.append((ids.copy(), x, x.copy()))
        z = x.reshape(len(ids), -1)
        scale = np.asarray(ids) + 1
        return (.5*scale*np.sum(z*z, axis=1), scale[:, None]*z if g else None,
                (scale[:, None, None]*np.eye(z.shape[1])[None]).reshape(-1,z.shape[1]) if h else None)
    expected, actual = BatchEnergyFixture(mapping, batch, 1), BatchEnergyFixture(mapping, batch, batch_size)
    x = np.arange(12, dtype=float)
    np.testing.assert_allclose(actual.eval_value(x), expected.eval_value(x))
    np.testing.assert_allclose(actual.eval_gradient(x), expected.eval_gradient(x))
    np.testing.assert_allclose(actual.eval_hessian(x).toarray(), expected.eval_hessian(x).toarray())
    np.testing.assert_allclose(actual.eval_projected_hessian(x).toarray(), expected.eval_projected_hessian(x).toarray())
    for ids, retained, copy in seen:
        assert len(ids) <= batch_size
        np.testing.assert_array_equal(retained, copy)
    assert set(np.concatenate([entry[0] for entry in seen])) == {0, 1, 2, 3}
    assert actual.max_feasible_step(x, x) == 1.


def test_batched_mesh_energy_errors_steps_and_recovery():
    from litepgo.dof import StencilDofMap
    mapping = StencilDofMap(2, [[0], [1]], 3)
    x = np.ones(6)
    def local(ids, x, g, h):
        return np.zeros(len(ids)), np.zeros((len(ids), 3)) if g else None, np.tile(np.eye(3), (len(ids), 1)) if h else None
    with pytest.raises(ValueError):
        BatchEnergyFixture(mapping, local, batch_size=0)
    with pytest.raises(RuntimeError, match='invalid batch outputs'):
        BatchEnergyFixture(mapping, lambda *args: (np.zeros(1), None, None)).eval_value(x)
    with pytest.raises(ValueError, match='Missing batch gradients'):
        BatchEnergyFixture(mapping, lambda *args: (np.zeros(2), None, None)).eval_gradient(x)
    def fail(*args):
        raise KeyError('batch failure')
    with pytest.raises(KeyError, match='batch failure'):
        BatchEnergyFixture(mapping, fail).eval_value(x)
    bounded = BatchEnergyFixture(mapping, local, step=lambda ids, x, dx: np.full(len(ids), .25))
    assert bounded.max_feasible_step(x, x) == .25
    with pytest.raises(RuntimeError, match='invalid batch steps'):
        BatchEnergyFixture(mapping, local, step=lambda ids, x, dx: np.full(len(ids), 2.)).max_feasible_step(x, x)
    assert bounded.eval_value(x) == 0.


def test_batched_mesh_energy_projection_empty_and_recovery():
    from litepgo.dof import StencilDofMap
    fail = True
    def batch(ids, x, g, h):
        if fail:
            raise ValueError('retry')
        return np.zeros(len(ids)), np.zeros((len(ids), 1)) if g else None, np.array([[-1.], [2.]]) if h else None
    obj = BatchEnergyFixture(StencilDofMap(1, [[0], [0]], 1), batch)
    with pytest.raises(ValueError, match='retry'):
        obj.eval_hessian(np.zeros(1))
    fail = False
    np.testing.assert_allclose(obj.eval_hessian(np.zeros(1)).toarray(), [[1.]])
    np.testing.assert_allclose(obj.eval_projected_hessian(np.zeros(1)).toarray(), [[2.]])
    empty = BatchEnergyFixture(StencilDofMap(1, [], 3), lambda *args: pytest.fail('empty batch invoked'))
    assert empty.eval_value(np.zeros(3)) == 0.
    np.testing.assert_array_equal(empty.eval_gradient(np.zeros(3)), np.zeros(3))


@pytest.mark.parametrize("kind", ["constant", "element", "vertex", "stencil"])
@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_mesh_energy_all_maps_share_batch_layout(kind, batch_size):
    from litepgo import dof, mesh
    topology = mesh.TetTopology(5, np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32))
    mapping = {
        "constant": lambda: dof.ConstantDofMap(topology, 2),
        "element": lambda: dof.ElementDofMap(topology, 2),
        "vertex": lambda: dof.VertexDofMap(topology, 2),
        "stencil": lambda: dof.StencilDofMap(5, [[0, 1], [1, 2, 3]], 2),
    }[kind]()
    counts = np.zeros(mapping.num_dofs)
    for e in range(mapping.num_elements):
        counts[mapping.dof_indices(e)] += 1

    def callback(ids, x, need_g, need_h):
        assert x.ndim == 2 and x.shape[0] == len(ids) <= batch_size
        h = np.tile(np.eye(x.shape[1]), (len(ids), 1)) if need_h else None
        return .5 * (x*x).sum(axis=1), x if need_g else None, h

    obj = BatchEnergyFixture(mapping, callback, batch_size=batch_size)
    x = np.arange(mapping.num_dofs, dtype=float)
    assert obj.eval_value(x) == .5 * (counts * x*x).sum()
    np.testing.assert_array_equal(obj.eval_gradient(x), counts * x)
    np.testing.assert_array_equal(obj.eval_hessian(x).toarray(), np.diag(counts))


def test_batch_none_is_infeasible_and_recovers():
    from litepgo import dof, energy
    mapping = dof.StencilDofMap(1, [[0]] * 32, 1)
    fail = True
    def callback(ids, x, need_g, need_h):
        if fail and 15 in ids:
            return None
        return (x[:, 0] ** 2, 2*x if need_g else None,
                np.full((len(ids), 1), 2.) if need_h else None)
    obj = BatchEnergyFixture(mapping, callback, batch_size=1)
    x = np.ones(1)
    for operation in (lambda: obj.eval_value(x), lambda: obj.eval_gradient(x), lambda: obj.eval_hessian(x)):
        with pytest.raises(ValueError, match="infeasible batch"):
            operation()
        fail = False
        operation()
        fail = True
    fail = False
    assert obj.eval_value(x) == 32
    np.testing.assert_array_equal(obj.eval_gradient(x), [64])
    np.testing.assert_array_equal(obj.eval_hessian(x).toarray(), [[64]])
