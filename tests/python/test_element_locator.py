import numpy as np
import pytest
from litepgo import fem, mesh


def test_extracted_surface_and_geometry_snapshot():
    X = np.array([[0., 0, 0], [.25, .25, .25], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cells = np.array([[1, 2, 3, 4], [0, 1, 3, 4], [0, 2, 1, 4], [0, 2, 3, 1]], dtype=np.int32)
    fields = [fem.InterpolatedField3D[fem.TetP1Basis](X[c]) for c in cells]
    locator = fem.ElementLocator[fem.TetP1Basis](fields)
    surface = mesh.extract_surface(mesh.TetMesh(X, cells))
    result = locator.locate(surface.positions)
    for p, e, xi in zip(surface.positions, result.element_ids, result.xi):
        np.testing.assert_allclose(fields[e].eval_value(xi), p, atol=1e-10)
    del fields
    np.testing.assert_array_equal(locator.locate(surface.positions).element_ids, result.element_ids)
    with pytest.raises(RuntimeError, match='point 0'):
        locator.locate(np.array([[2., 2, 2]]))
    with pytest.raises(ValueError):
        locator.locate(np.array([[np.nan, 0, 0]]))
    assert locator.locate(np.empty((0, 3))).xi.shape == (0, 3)


def test_hex_geometry_for_hermite_displacement():
    X = np.array([fem.HexQ1Basis.reference_node(i) for i in range(8)])
    X[6] += [.3, 0, .4]
    field = fem.InterpolatedField3D[fem.HexQ1Basis](X)
    locator = fem.ElementLocator[fem.HexQ1Basis]([field])
    xi = np.array([[.2, .3, .4], [0, .5, 1]])
    points = np.array([field.eval_value(p) for p in xi])
    np.testing.assert_allclose(locator.locate(points).xi, xi, atol=1e-8)
