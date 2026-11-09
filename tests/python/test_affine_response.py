import numpy as np
import pytest
import scipy.sparse as sp
from litepgo import energy


@pytest.mark.parametrize("convert", [np.asarray, sp.csc_matrix, sp.csr_matrix, sp.coo_matrix, sp.csc_array])
def test_values_derivatives_snapshots_and_export(convert):
    assert "AffineResponse" in energy.__all__
    A = np.array([[1., 0., -2.], [.5, 3., 0.]])
    source = convert(A.copy())
    offset = np.array([4., -1.])
    r = energy.AffineResponse(A=source, offset=offset)
    x = np.array([1., 2., -3.])
    dx, w = np.array([.2, -.5, .7]), np.array([2., -3.])
    offset[:] = np.nan
    if sp.issparse(source):
        source.data[:] = np.nan
    else:
        source[:] = np.nan
    assert isinstance(r, energy.Response)
    assert (r.num_inputs, r.num_outputs) == (3, 2)
    r_evaluation = r.evaluate(x)
    J = r_evaluation.jacobian()
    value = r_evaluation.value
    np.testing.assert_allclose(value, [11., 5.5])
    np.testing.assert_array_equal(J.toarray(), A)
    np.testing.assert_allclose(r.evaluate(x).jvp(dx), A @ dx)
    np.testing.assert_allclose(r.evaluate(x).vjp(w), A.T @ w)
    H = r.evaluate(x).hessian(weights=w)
    assert H.shape == (3, 3) and H.nnz == 0
    assert r.evaluate(x).hessian(w).nnz == 0
    J.data[:] = 99
    np.testing.assert_array_equal(r.evaluate(x).jacobian().toarray(), A)
    np.testing.assert_allclose(r.evaluate(-x).value, -A @ x + [4.0, -1.0])


def test_duplicate_sparse_entries_stored_zeros_and_strided_input():
    A = sp.coo_matrix(([1., 2., 0., -1.], ([0, 0, 1, 1], [0, 0, 1, 2])), shape=(2, 3))
    r = energy.AffineResponse(A)
    x = np.arange(6.)[::2]
    J = r.evaluate(x).jacobian()
    np.testing.assert_array_equal(J.toarray(), [[3., 0., 0.], [0., 0., -1.]])
    assert J.has_canonical_format and J.nnz == 3
    np.testing.assert_array_equal(r.evaluate(x).value, [0.0, -4.0])
    assert A.nnz == 4  # Conversion does not mutate the caller's sparse matrix.
    dense = np.arange(12.).reshape(3, 4).T[::2]
    np.testing.assert_allclose(energy.AffineResponse(dense).evaluate(x).value, dense @ x)


@pytest.mark.parametrize("offset", [None, np.empty(0)])
def test_zero_offset_identity_and_rank_deficient_maps(offset):
    x = np.array([1., -2., 3.])
    np.testing.assert_array_equal(energy.AffineResponse(np.eye(3), offset).evaluate(x).value, x)
    np.testing.assert_array_equal(energy.AffineResponse(np.ones((2, 3)), offset).evaluate(x).value, [2.0, 2.0])
    constant = energy.AffineResponse(np.zeros((2, 3)), np.array([2., -1.]))
    np.testing.assert_array_equal(constant.evaluate(x).value, [2.0, -1.0])
    assert constant.evaluate(x).jacobian().nnz == 0
    np.testing.assert_array_equal(constant.evaluate(x).jvp(x), 0)


@pytest.mark.parametrize("m,n", [(0, 0), (0, 3), (2, 0)])
def test_empty_spaces(m, n):
    r = energy.AffineResponse(np.zeros((m, n)), np.ones(m))
    x = np.zeros(n)
    np.testing.assert_array_equal(r.evaluate(x).value, np.ones(m))
    assert r.evaluate(x).jacobian().shape == (m, n)
    assert r.evaluate(x).hessian(np.ones(m)).shape == (n, n)
    np.testing.assert_array_equal(r.evaluate(x).jvp(x), np.zeros(m))
    np.testing.assert_array_equal(r.evaluate(x).vjp(np.ones(m)), np.zeros(n))


def test_composition_exact_hessian_and_finite_differences():
    A = np.array([[1., 0., -2.], [.5, 3., 0.]])
    b = np.array([.1, -.2])
    affine = energy.AffineResponse(A, b)
    r = energy.ComposedResponse(energy.ExpResponse(2), affine)
    x, w = np.array([-.3, .1, .2]), np.array([.4, -.7])
    y = np.exp(A @ x + b)
    H = r.evaluate(x).hessian(w).toarray()
    np.testing.assert_allclose(H, A.T @ np.diag(w * y) @ A)
    columns = [(r.evaluate(x + d).vjp(w) - r.evaluate(x - d).vjp(w)) / 2e-6 for d in 1e-6 * np.eye(3)]
    np.testing.assert_allclose(H, np.column_stack(columns), atol=1e-9, rtol=1e-7)
    outer = energy.ComposedResponse(affine, energy.ExpResponse(3))
    np.testing.assert_allclose(outer.evaluate(x).hessian(w).toarray(), np.diag(A.T @ w * np.exp(x)))


def test_invalid_inputs_and_overflow():
    for bad in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError):
            energy.AffineResponse([[bad]])
        with pytest.raises(ValueError):
            energy.AffineResponse(np.eye(2), np.array([1., bad]))
    with pytest.raises(ValueError):
        energy.AffineResponse(np.ones((2, 3)), np.zeros(3))
    with pytest.raises(TypeError):
        energy.AffineResponse(np.ones(3))
    r = energy.AffineResponse(np.ones((2, 3)))
    for x in [np.zeros(2), np.full(3, np.nan)]:
        with pytest.raises(ValueError):
            r.evaluate(x).value
    with pytest.raises(ValueError):
        r.evaluate(np.zeros(3)).hessian(np.ones(3))
    with pytest.raises(ValueError):
        r.evaluate(np.zeros(3)).hessian(np.full(2, np.inf))
    huge = energy.AffineResponse([[np.finfo(float).max]])
    for evaluate in [lambda: huge.evaluate(np.array([2.0])).value,
                     lambda: huge.evaluate(np.zeros(1)).jvp(np.array([2.0])),
                     lambda: huge.evaluate(np.zeros(1)).vjp(np.array([2.0]))]:
        with pytest.raises(ValueError, match="non-finite"):
            evaluate()
    np.testing.assert_array_equal(huge.evaluate(np.zeros(1)).value, 0)


@pytest.mark.parametrize("size", [0, 3])
def test_identity_response(size):
    assert "IdentityResponse" in energy.__all__
    r = energy.IdentityResponse(size=size)
    x = np.arange(size, dtype=float)
    r_evaluation = r.evaluate(x)
    J = r_evaluation.jacobian()
    value = r_evaluation.value
    np.testing.assert_array_equal(value, x)
    np.testing.assert_array_equal(J.toarray(), np.eye(size))
    np.testing.assert_array_equal(r.evaluate(x).jvp(x + 1), x + 1)
    np.testing.assert_array_equal(r.evaluate(x).vjp(x - 1), x - 1)
    H = r.evaluate(x).hessian(x)
    assert r.has_hessian and H.shape == (size, size) and H.nnz == 0


@pytest.mark.parametrize("num_inputs,num_outputs", [(3, 2), (0, 2), (3, 0)])
def test_constant_response(num_inputs, num_outputs):
    assert "ConstantResponse" in energy.__all__
    value = np.arange(1., num_outputs + 1)
    r = energy.ConstantResponse(num_inputs=num_inputs, value=value)
    value[:] = np.nan
    x = np.ones(num_inputs)
    r_evaluation = r.evaluate(x)
    J = r_evaluation.jacobian()
    result = r_evaluation.value
    np.testing.assert_array_equal(result, np.arange(1., num_outputs + 1))
    assert J.shape == (num_outputs, num_inputs) and J.nnz == 0
    w = np.ones(num_outputs)
    np.testing.assert_array_equal(r.evaluate(x).jvp(x), np.zeros(num_outputs))
    np.testing.assert_array_equal(r.evaluate(x).vjp(w), np.zeros(num_inputs))
    H = r.evaluate(x).hessian(w)
    assert r.has_hessian and H.shape == (num_inputs, num_inputs) and H.nnz == 0


def test_identity_and_constant_response_validation():
    with pytest.raises(ValueError):
        energy.IdentityResponse(-1)
    with pytest.raises(ValueError):
        energy.ConstantResponse(-1, np.empty(0))
    with pytest.raises(ValueError):
        energy.ConstantResponse(3, np.array([np.nan]))
    with pytest.raises(ValueError):
        energy.IdentityResponse(1).evaluate(np.zeros(1)).jvp(np.empty(0))
    with pytest.raises(ValueError):
        energy.ConstantResponse(1, np.empty(0)).evaluate(np.zeros(1)).vjp(np.ones(1))
