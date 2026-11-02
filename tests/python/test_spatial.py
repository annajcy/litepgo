import numpy as np
import pytest
from litepgo import spatial, fem


def test_spatial_snapshot_queries_and_filters():
    boxes = [spatial.AABB([0,0,0],[2,2,2]),spatial.AABB([1,1,1],[3,3,3]),spatial.AABB([5,5,5],[6,6,6])]
    index = spatial.SpatialHash(boxes,1)
    boxes.clear()
    assert sorted(index.query(spatial.AABB([2,2,2],[2,2,2]))) == [0,1]
    assert index.overlapping_pairs() == [[0,1]]
    assert index.overlapping_pairs(lambda i,j: False) == []
    def fail(i,j):
        raise RuntimeError('callback failed')
    with pytest.raises(RuntimeError,match='callback failed'):
        index.overlapping_pairs(fail)
    with pytest.raises(TypeError):
        index.overlapping_pairs(3)
    with pytest.raises(ValueError):
        spatial.SpatialHash([],0)
    with pytest.raises(ValueError):
        index.query(spatial.AABB([1,0,0],[0,0,0]))
    with pytest.raises(OverflowError):
        index.query(spatial.AABB([np.inf]*3,[np.inf]*3))
    assert spatial.SpatialHash([]).overlapping_pairs() == []


def test_point_embedding_constructor():
    ids = np.array([1,0],dtype=np.int32)
    xi = np.array([[.1,.2,.3],[0,0,0.]])
    embedding = fem.PointEmbedding(ids,xi)
    ids[:] = 4; xi[:] = 5
    np.testing.assert_array_equal(embedding.element_ids,[1,0])
    np.testing.assert_allclose(embedding.xi[0],[.1,.2,.3])
    for bad_ids,bad_xi in [([-1],[[0.,0,0]]),([0],[[np.nan,0,0]]),([0],np.zeros((2,3))),([0],np.zeros((1,2)))]:
        with pytest.raises(ValueError):
            fem.PointEmbedding(np.array(bad_ids,dtype=np.int32),np.array(bad_xi))
    assert fem.PointEmbedding(np.empty(0,dtype=np.int32),np.empty((0,3))).xi.shape == (0,3)
