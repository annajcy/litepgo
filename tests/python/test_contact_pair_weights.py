import numpy as np
import pytest

from litepgo import contact, mesh


def surface():
    return mesh.TriangleMesh(np.array([[0., 0, 0], [10, 0, 0], [0, 10, 0],
                                       [2, 2, .5], [6, 2, 4.5], [2, 6, 4.5]]),
                             np.array([[0, 1, 2], [3, 4, 5]]))


def uniform(s, scale):
    return contact.ContactPairWeights([0] * len(s.positions), [0] * len(contact.contact_edges(s.topology)),
                                       [0] * len(s.cells), [[scale]])


@pytest.fixture(params=["native", "toolkit"])
def make(request):
    if request.param == "toolkit" and not hasattr(contact, "IPCTKContactEnergy"):
        pytest.skip("toolkit backend disabled")

    def build(s, pair_weights=None, area=True):
        kwargs = dict(dhat=.7, min_distance=.05, use_area_weighting=area)
        if request.param == "native":
            return contact.IPCContactEnergy(s, contact.IPCContactOptions(**kwargs), contact.SpatialHash(), pair_weights)
        return contact.IPCTKContactEnergy(s, contact.IPCTKContactOptions(**kwargs), pair_weights)
    return build


def test_contact_edge_numbering():
    expected = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]]
    for faces in ([[2, 1, 0], [0, 3, 2]], [[2, 3, 0], [0, 1, 2]]):
        topology = mesh.TriangleTopology(4, np.array(faces))
        np.testing.assert_array_equal(contact.contact_edges(topology), expected)
    assert contact.contact_edges(mesh.TriangleTopology(0, np.empty((0, 3), dtype=int))).shape == (0, 2)


@pytest.mark.parametrize("area", [False, True])
@pytest.mark.parametrize("scale", [1, .5, 2])
def test_uniform_table_scales_energy_derivatives_and_snapshot(make, area, scale):
    s = surface()
    base, weighted = make(s, area=area), make(s, uniform(s, scale), area)
    x = s.positions.ravel()
    for e in (base, weighted):
        e.update_pattern(x)
    assert weighted.eval_value(x) == pytest.approx(scale * base.eval_value(x), rel=1e-12)
    np.testing.assert_allclose(weighted.eval_gradient(x), scale * base.eval_gradient(x), atol=1e-12)
    np.testing.assert_allclose(weighted.eval_hessian(x).toarray(), scale * base.eval_hessian(x).toarray(), atol=1e-11)
    np.testing.assert_allclose(weighted.eval_projected_hessian(x).toarray(),
                               scale * base.eval_projected_hessian(x).toarray(), atol=1e-10)
    a, b = base.contact_snapshot(x), weighted.contact_snapshot(x)
    assert a.types == b.types and len(b.types) == weighted.num_active_pairs(x)
    np.testing.assert_array_equal(a.vertex_ids, b.vertex_ids)
    np.testing.assert_allclose(b.weights, scale * a.weights, rtol=1e-12)
    np.testing.assert_array_equal(a.mollifiers, b.mollifiers)


@pytest.mark.parametrize("area", [False, True])
def test_parent_multipliers_are_applied_before_vertex_vertex_merging(make, area):
    p = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0], [-.2, -.3, .4], [-1, -2, 1], [-2, -1, 1]])
    s = mesh.TriangleMesh(p, np.array([[0, 1, 2], [3, 4, 5]]))
    table = np.zeros((4, 4))
    table[0, 2] = table[2, 0] = .5
    table[1, 1] = 2
    vertices, edges, faces = np.array([0, 3, 3, 1, 3, 3]), np.full(6, 3), np.array([1, 2])
    config = contact.ContactPairWeights(vertices, edges, faces, table)
    e = make(s, config, area)
    # Construction copies every input; returned configuration properties are also copies.
    vertices[:] = edges[:] = faces[:] = 999
    table[:] = 0
    config.weights[:] = 0
    config.vertex_groups[:] = [999] * 6
    x = p.ravel().copy()
    snapshot = e.contact_snapshot(x)
    assert snapshot.types == ("VV",)
    np.testing.assert_array_equal(snapshot.vertex_ids, [[0, 3, -1, -1]])
    areas = np.linalg.norm(np.cross(p[s.cells[:, 1]] - p[s.cells[:, 0]],
                                    p[s.cells[:, 2]] - p[s.cells[:, 0]]), axis=1) / 2
    weight = (.5 * areas[0] + 2 * areas[1]) / 12 if area else 2.5
    assert snapshot.weights[0] == pytest.approx(weight, abs=1e-14)
    d = np.sum((p[0] - p[3])**2) - .05**2
    h = .7 * (.7 + .1)
    assert e.eval_value(x) == pytest.approx(-weight * (d - h)**2 * np.log(d / h), rel=1e-12)
    basis = np.eye(len(x)) * 1e-6
    fd_gradient = [(e.eval_value(x + dx) - e.eval_value(x - dx)) / 2e-6 for dx in basis]
    np.testing.assert_allclose(e.eval_gradient(x), fd_gradient, atol=1e-9)
    e.update_pattern(x)
    fd_hessian = np.column_stack([(e.eval_gradient(x + dx) - e.eval_gradient(x - dx)) / 2e-6 for dx in basis])
    np.testing.assert_allclose(e.eval_hessian(x).toarray(), fd_hessian, atol=1e-8)
    reordered = mesh.TriangleMesh(p, s.cells[::-1, ::-1].copy())
    relabeled = contact.ContactPairWeights(config.vertex_groups, config.edge_groups,
                                           config.face_groups[::-1], config.weights)
    other = make(reordered, relabeled, area)
    assert other.eval_value(x) == pytest.approx(e.eval_value(x), rel=1e-12)
    np.testing.assert_allclose(other.eval_gradient(x), e.eval_gradient(x), atol=1e-12)


def test_zero_removes_pairs_and_ccd_while_global_intersections_remain_invalid(make):
    s = surface()
    base, zero = make(s), make(s, uniform(s, 0))
    x = s.positions.ravel().copy()
    dx = np.zeros_like(x)
    dx[11] = -2
    step = base.max_feasible_step(x, dx)
    assert 0 < step < 1
    assert zero.max_feasible_step(x, dx) == 1
    for scale in (.001, .5, 2):
        assert make(s, uniform(s, scale)).max_feasible_step(x, dx) == step
    assert zero.eval_value(x) == 0
    np.testing.assert_array_equal(zero.eval_gradient(x), np.zeros_like(x))
    assert zero.num_active_pairs(x) == 0 and zero.min_active_distance(x) == float("inf")
    assert zero.contact_snapshot(x).types == ()
    assert not zero.update_pattern(x)
    assert zero.eval_hessian(x).nnz == 0
    x[11] = .01  # Below min_distance, still excluded from this contact item.
    assert zero.eval_value(x) == 0 and zero.max_feasible_step(x, np.zeros_like(x)) == 1
    x[11] = -.5  # An edge now crosses a face: the independent global check still rejects it.
    with pytest.raises(ValueError, match="intersect"):
        zero.max_feasible_step(x, np.zeros_like(x))
    x = x.reshape(-1, 3)
    x[3:] = [[2, 2, 0], [3, 2, 0], [2, 3, 0]]  # Coplanar overlap, with no edge crossing the face plane.
    with pytest.raises(ValueError, match="intersect"):
        zero.max_feasible_step(x.ravel(), np.zeros(x.size))


@pytest.mark.parametrize("bad", ["vertices_length", "edges_length", "faces_length", "negative_id", "large_id",
                                "empty", "nonsquare", "asymmetric", "negative", "nan", "inf"])
def test_invalid_pair_weights_rejected_when_constructing_energy(make, bad):
    s = surface()
    v, e, f, w = [0]*6, [0]*6, [0]*2, np.ones((3, 3))
    if bad.endswith("length"):
        {"vertices_length": v, "edges_length": e, "faces_length": f}[bad].pop()
    elif bad == "negative_id":
        v[0] = -1
    elif bad == "large_id":
        e[0] = 3
    elif bad == "empty":
        w = np.empty((0, 0))
    elif bad == "nonsquare":
        w = np.ones((2, 3))
    elif bad == "asymmetric":
        w[0, 1] = .5
    else:
        w[0, 0] = {"negative": -1, "nan": np.nan, "inf": np.inf}[bad]
    with pytest.raises(ValueError, match="ContactPairWeights"):
        make(s, contact.ContactPairWeights(v, e, f, w))


def test_noninteger_group_ids_rejected_at_python_boundary():
    with pytest.raises(TypeError):
        contact.ContactPairWeights([.5]*6, [0]*6, [0]*2, [[1.]])


@pytest.mark.parametrize("table", [np.ones(3), np.ones((2, 2, 2)), np.array([[1+2j]])])
def test_table_dimensions_and_complex_values_rejected_at_python_boundary(table):
    with pytest.raises(ValueError, match="ContactPairWeights"):
        contact.ContactPairWeights([0]*6, [0]*6, [0]*2, table)


def test_unrepresentable_weighted_contribution_is_rejected(make):
    s = surface()
    energy = make(s, uniform(s, np.finfo(float).max))
    with pytest.raises(ValueError, match="unrepresentable"):
        energy.contact_snapshot(s.positions.ravel())


@pytest.mark.skipif(not hasattr(contact, "IPCTKContactEnergy"), reason="toolkit backend disabled")
@pytest.mark.parametrize("area", [False, True])
@pytest.mark.parametrize("p", [
    [[0, 0, 0], [2, 0, 0], [0, 2, 0], [-.2, -.3, .4], [-1, -2, 1], [-2, -1, 1]],
    [[0, 0, 0], [2, 0, 0], [0, 2, 0], [.7, -.2, .4], [2, -2, 1], [-2, -1, 1]],
    [[0, 0, 0], [2, 0, 0], [0, 2, 0], [.4, .3, .4], [2, 2, 1], [-1, 2, 1]],
    [[-1, 0, 0], [1, 0, 0], [0, -4, -2], [-.7, -.004, .2], [.8, .008, .2], [0, 4, 2]],
    [[-1, 0, 0], [1, 0, 0], [0, -4, -2], [-.7, 0, .2], [.8, 0, .2], [0, 4, 2]],
])
def test_mixed_groups_match_toolkit_including_mollified_features(p, area):
    s = mesh.TriangleMesh(np.array(p, dtype=float), np.array([[0, 1, 2], [3, 4, 5]]))
    table = np.array([[1, .5, 0, 2], [.5, 2, .7, 0], [0, .7, 1, .3], [2, 0, .3, 1]])
    v, e, f = np.array([0, 1, 3, 2, 0, 1]), np.arange(6) % 4, np.array([2, 3])
    pw = contact.ContactPairWeights(v, e, f, table)
    permutation = np.array([3, 1, 0, 2])
    permuted_table = np.empty_like(table)
    permuted_table[np.ix_(permutation, permutation)] = table
    relabeled = contact.ContactPairWeights(permutation[v], permutation[e], permutation[f], permuted_table)
    options = dict(dhat=.7, min_distance=.05, use_area_weighting=area)
    native = contact.IPCContactEnergy(s, contact.IPCContactOptions(**options), contact.SpatialHash(), pw)
    toolkit = contact.IPCTKContactEnergy(s, contact.IPCTKContactOptions(**options), pw)
    renamed = contact.IPCContactEnergy(s, contact.IPCContactOptions(**options), contact.SpatialHash(), relabeled)
    x = s.positions.ravel()
    for other in (toolkit, renamed):
        assert other.eval_value(x) == pytest.approx(native.eval_value(x), rel=1e-11, abs=1e-13)
        np.testing.assert_allclose(other.eval_gradient(x), native.eval_gradient(x), rtol=1e-8, atol=1e-10)
        for energy in (native, other):
            energy.update_pattern(x)
        np.testing.assert_allclose(other.eval_hessian(x).toarray(), native.eval_hessian(x).toarray(), rtol=1e-7, atol=1e-8)
        assert other.num_active_pairs(x) == native.num_active_pairs(x)
        assert other.min_active_distance(x) == native.min_active_distance(x)
        assert sum(other.contact_snapshot(x).mollifiers == 0) == sum(native.contact_snapshot(x).mollifiers == 0)
