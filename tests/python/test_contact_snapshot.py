import gc
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from litepgo import contact, mesh

EXAMPLES = Path(__file__).resolve().parents[2] / "examples/contact"
spec = importlib.util.spec_from_file_location("snapshot_example", EXAMPLES / "contact_snapshot.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.fixture(params=["native", "toolkit"])
def make_barrier(request):
    if request.param == "toolkit" and not hasattr(contact, "IPCTKContactEnergy"):
        pytest.skip("toolkit backend disabled")

    def make(surface, weighted=True, physical=False):
        kwargs = dict(dhat=.7, min_distance=.05, use_area_weighting=weighted, use_physical_barrier=physical)
        if request.param == "native":
            return contact.IPCContactEnergy(surface, contact.IPCContactOptions(**kwargs), contact.SpatialHash())
        return contact.IPCTKContactEnergy(surface, contact.IPCTKContactOptions(**kwargs))
    return make


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("physical", [False, True])
def test_snapshot_contains_complete_assembly_records(make_barrier, weighted, physical):
    surface = example.contact_scene()
    barrier = make_barrier(surface, weighted, physical)
    x = surface.positions.copy().reshape(5, 6, 3)
    x[:, 3:, 2] -= .05
    x = x.ravel()
    before = barrier.eval_value(x)
    snapshot = barrier.contact_snapshot(x)
    assert isinstance(snapshot, contact.ContactSnapshot)
    count = len(snapshot.types)
    assert count == barrier.num_active_pairs(x)
    assert snapshot.vertex_ids.shape == (count, 4)
    assert all(getattr(snapshot, field).shape == (count,) for field in ("distances", "weights", "mollifiers"))
    np.testing.assert_array_equal(snapshot.positions.ravel(), x)
    assert set(snapshot.types) == {"VV", "VE", "VF", "EE"}
    assert snapshot.distances.min() == barrier.min_active_distance(x)
    assert np.all(snapshot.weights > 0)
    assert np.all((snapshot.mollifiers >= 0) & (snapshot.mollifiers <= 1))
    assert np.any((snapshot.mollifiers > 0) & (snapshot.mollifiers < 1))
    assert np.any(snapshot.mollifiers == 0)
    for kind, ids, mollifier in zip(snapshot.types, snapshot.vertex_ids, snapshot.mollifiers):
        n = {"VV": 2, "VE": 3, "VF": 4, "EE": 4}[kind]
        assert np.all((ids[:n] >= 0) & (ids[:n] < len(surface.positions)))
        assert np.all(ids[n:] == -1)
        assert len(set(ids[:n])) == n
        if kind != "EE":
            assert mollifier == 1
    # Snapshot weights contain geometry and group factors, independently of normalization.
    s = snapshot.distances**2 - .05**2
    h = .7 * (.7 + 2 * .05)
    reconstructed = np.sum(snapshot.weights * snapshot.mollifiers * -(s - h)**2 * np.log(s / h))
    if physical:
        reconstructed *= .7 / h**2
    assert reconstructed == pytest.approx(before, rel=1e-12)
    assert barrier.eval_value(x) == before
    pattern = barrier.eval_hessian(x)
    far = x.reshape(5, 6, 3).copy()
    far[:, 3:, 2] += 20
    empty = barrier.contact_snapshot(far.ravel())
    assert empty.types == ()
    assert empty.vertex_ids.shape == (0, 4)
    assert empty.distances.shape == empty.weights.shape == empty.mollifiers.shape == (0,)
    assert barrier.min_active_distance(far.ravel()) == float("inf")
    np.testing.assert_array_equal(barrier.eval_hessian(x).indices, pattern.indices)
    np.testing.assert_array_equal(barrier.eval_hessian(x).indptr, pattern.indptr)

@pytest.mark.parametrize("empty", [False, True])
def test_snapshot_is_readonly_and_owns_its_data(make_barrier, empty):
    surface = example.contact_scene()
    barrier = make_barrier(surface)
    x = surface.positions.copy().reshape(5, 6, 3)
    if empty:
        x[:, 3:, 2] += 20
    snapshot = barrier.contact_snapshot(x.ravel())
    arrays = {name: getattr(snapshot, name) for name in
              ("positions", "vertex_ids", "distances", "weights", "mollifiers")}
    saved = {name: array.copy() for name, array in arrays.items()}
    assert isinstance(snapshot.types, tuple)
    for name, array in arrays.items():
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array[...] = 0
        with pytest.raises(AttributeError):
            setattr(snapshot, name, array)
    with pytest.raises(AttributeError):
        snapshot.types = ()
    x += 1
    barrier.contact_snapshot(x.ravel())
    del snapshot, barrier, x, surface
    gc.collect()
    for name, array in arrays.items():
        np.testing.assert_array_equal(array, saved[name])


def test_snapshot_validates_input(make_barrier):
    surface = example.contact_scene()
    barrier = make_barrier(surface)
    with pytest.raises(ValueError):
        barrier.contact_snapshot(np.zeros(3))
    x = surface.positions.ravel().copy()
    x[0] = np.nan
    with pytest.raises(ValueError):
        barrier.contact_snapshot(x)


def test_mollified_ee_keeps_four_vertices_for_a_vertex_edge_closest_feature(make_barrier):
    positions = np.array([[-1, 0, 0], [1, 0, 0], [0, -4, -2],
                          [0, .004, .2], [2, .008, .2], [0, 4, 2]])
    surface = mesh.TriangleMesh(positions, np.array([[0, 1, 2], [3, 4, 5]]))
    snapshot = make_barrier(surface).contact_snapshot(positions.ravel())
    rows = [i for i, kind in enumerate(snapshot.types) if kind == "EE" and snapshot.mollifiers[i] < 1]
    assert rows
    assert any(set(snapshot.vertex_ids[i]) == {0, 1, 3, 4} for i in rows)
