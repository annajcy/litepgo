import numpy as np
import pytest
from litepgo import energy, fem

BASES = [
    (fem.LinearLagrangeBasis, 1), (fem.CubicHermiteBasis, 1),
    (fem.TriangleP1Basis, 2), (fem.QuadHermiteBasis, 2),
    (fem.TetP1Basis, 3), (fem.HexQ1Basis, 3), (fem.HexHermiteBasis, 3),
]
FACTORIES = [
    fem.value_sampling_response,
    fem.gradient_sampling_response,
    fem.hessian_sampling_response,
]


@pytest.mark.parametrize("basis,d", BASES)
@pytest.mark.parametrize("order", range(3))
def test_sampling_matches_spatial_response(basis, d, order):
    m = 2
    C = np.random.default_rng(42).normal(size=(basis.num_basis_functions, m))
    c = C.ravel()
    # Multiple points, including polynomial extrapolation outside the domain.
    points = np.array([np.linspace(.2, .4, d), np.linspace(1.1, 1.3, d)])
    saved = points.copy()
    factory = FACTORIES[order][basis]
    assert factory.__module__ == "litepgo._core.fem"
    assert basis.__name__ in factory.__qualname__
    sample = factory(points=points, components=m)
    assert type(sample) is energy.AffineResponse
    points[:] = 0
    spatial = fem.InterpolatedResponse[basis](C)
    expected = []
    for xi in saved:
        if order == 0:
            expected.extend(spatial.eval_value(xi))
        elif order == 1:
            expected.extend(spatial.eval_jacobian(xi).toarray().ravel(order="F"))
        else:
            component_hessians = [spatial.eval_hessian(xi, w).toarray() for w in np.eye(m)]
            for i in range(d):
                for j in range(i, d):
                    expected.extend(H[i, j] for H in component_hessians)
    value, A = sample.eval_value_and_jacobian(c)
    assert sample.num_inputs == len(c) and sample.num_outputs == len(expected)
    np.testing.assert_allclose(value, expected, atol=1e-12)
    np.testing.assert_allclose(A @ c, expected, atol=1e-12)
    v = np.linspace(-.3, .8, len(c))
    w = np.linspace(-1., .7, len(value))
    eps = 1e-5
    numerical = (sample.eval_value(c + eps*v) - sample.eval_value(c - eps*v))/(2*eps)
    np.testing.assert_allclose(sample.eval_jvp(c, v), numerical, atol=1e-8)
    np.testing.assert_allclose(sample.eval_vjp(c, w), A.T @ w)
    assert sample.has_hessian
    H = sample.eval_hessian(c, w)
    assert H.shape == (len(c), len(c)) and H.nnz == 0
    original = value.copy()
    sample.eval_value(-c)
    np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("factory", FACTORIES)
def test_validation_and_empty_points(factory):
    make = factory[fem.TetP1Basis]
    for points in [np.zeros((2, 2)), np.full((1, 3), np.nan), np.full((1, 3), np.inf)]:
        with pytest.raises(ValueError):
            make(points, components=2)
    for components in [0, -1, np.iinfo(np.int32).max]:
        with pytest.raises(ValueError):
            make(np.empty((0, 3)), components=components)
    empty = make(np.empty((0, 3)), components=2)
    c = np.ones(8)
    assert empty.eval_value(c).shape == (0,)
    assert empty.eval_jacobian(c).shape == (0, 8)
    np.testing.assert_array_equal(empty.eval_vjp(c, np.empty(0)), np.zeros(8))
    assert empty.eval_hessian(c, np.empty(0)).nnz == 0
    with pytest.raises(ValueError):
        empty.eval_value(np.zeros(7))
    with pytest.raises(ValueError):
        empty.eval_value(np.full(8, np.nan))


def test_zero_spatial_hessian_and_composition():
    sample = fem.hessian_sampling_response[fem.TetP1Basis](np.zeros((2, 3)), components=2)
    assert sample.num_outputs == 24  # two points * six independent entries * two components
    assert sample.eval_jacobian(np.ones(8)).nnz == 0
    np.testing.assert_array_equal(sample.eval_value(np.ones(8)), np.zeros(24))
    value = fem.value_sampling_response[fem.LinearLagrangeBasis](np.array([[.25], [.75]]), components=1)
    composed = energy.ComposedResponse(energy.ExpResponse(2), value)
    c = np.array([.1, -.2])
    A = value.eval_jacobian(c).toarray()
    w = np.array([1., -2.])
    np.testing.assert_allclose(composed.eval_hessian(c, w).toarray(),
                               A.T @ np.diag(w*np.exp(A @ c)) @ A)


def test_overflow_and_1d_points_shape():
    make = fem.hessian_sampling_response[fem.CubicHermiteBasis]
    with pytest.raises(ValueError, match="non-finite"):
        make(np.array([[np.finfo(float).max]]), components=1)
    with pytest.raises((ValueError, TypeError)):
        make(np.array([.2, .3]), components=1)


def test_template_registry_keeps_classes_and_factories_distinct():
    spatial = fem.InterpolatedResponse[fem.LinearLagrangeBasis](np.ones((2, 1)))
    assert isinstance(spatial, fem.InterpolatedResponse)
    assert issubclass(type(spatial), fem.InterpolatedResponse)
    for factory in FACTORIES:
        sample = factory[fem.LinearLagrangeBasis](np.zeros((1, 1)), components=1)
        assert not isinstance(sample, factory)
        assert not issubclass(type(sample), factory)
        assert fem.LinearLagrangeBasis in [key[0] for key in factory.instances]
