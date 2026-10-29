import sys

import numpy as np
import pytest
import scipy.sparse as sp

import litepgo.linalg.sparse_solver as la

HAS_ACCELERATE = hasattr(la, "AccelerateLLT")
HAS_MKL = hasattr(la, "PardisoLLT")
ALL_SOLVERS = ([la.EigenLDLT, la.EigenSparseLU] + ([la.AccelerateLLT, la.AccelerateLDLT] if HAS_ACCELERATE else []) +
               ([la.PardisoLLT, la.PardisoLDLT] if HAS_MKL else []))
INDEFINITE_SOLVERS = ([la.EigenSparseLU] + ([la.AccelerateLDLT] if HAS_ACCELERATE else []) +
                      ([la.PardisoLDLT] if HAS_MKL else []))
CHOLESKY_SOLVERS = ([la.AccelerateLLT] if HAS_ACCELERATE else []) + ([la.PardisoLLT] if HAS_MKL else [])


def laplacian(n, scale=1.0):
    return sp.diags([-scale, 2.0 * scale, -scale], [-1, 0, 1], shape=(n, n), format="csc")


def saddle_point(n):
    ones = sp.csc_matrix(np.ones((1, n)))
    return sp.bmat([[laplacian(n), ones.T], [ones, None]], format="csc")


@pytest.mark.parametrize("solver_type", ALL_SOLVERS)
def test_solves_positive_definite(solver_type):
    A = laplacian(30)
    rhs = np.linspace(-1.0, 2.0, 30)
    solver = solver_type()
    assert solver.compute(A)
    np.testing.assert_allclose(solver.solve(rhs), np.linalg.solve(A.toarray(), rhs), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("solver_type", INDEFINITE_SOLVERS)
def test_solves_saddle_point(solver_type):
    A = saddle_point(10)
    rhs = np.arange(11, dtype=float)
    solver = solver_type()
    assert solver.compute(A)
    np.testing.assert_allclose(solver.solve(rhs), np.linalg.solve(A.toarray(), rhs), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("solver_type", CHOLESKY_SOLVERS)
def test_cholesky_fails_on_indefinite(solver_type):
    solver = solver_type()
    assert not solver.compute(saddle_point(10))
    with pytest.raises(RuntimeError):
        solver.solve(np.ones(11))


@pytest.mark.parametrize(
    "convert",
    [sp.csr_matrix, sp.coo_matrix, sp.csc_array, lambda A: A.toarray(), lambda A: sp.csc_matrix(
        (A.data, A.indices.astype(np.int64), A.indptr.astype(np.int64)), shape=A.shape)],
    ids=["csr", "coo", "csc_array", "dense", "int64_indices"])
def test_accepts_other_matrix_formats(convert):
    A = laplacian(8)
    solver = la.EigenLDLT()
    assert solver.compute(convert(A))
    np.testing.assert_allclose(A @ solver.solve(np.ones(8)), np.ones(8), atol=1e-12)


def test_sums_duplicates_and_sorts_without_touching_the_input():
    # scipy defines duplicate entries as summed, so [[2, 0], [0, 2]] here.
    duplicated = sp.csc_matrix((np.array([1.0, 1.0, 2.0]), np.array([0, 0, 1]), np.array([0, 2, 3])), shape=(2, 2))
    # Column 0 lists row 1 before row 0: [[3, 0], [1, 4]].
    unsorted = sp.csc_matrix((np.array([1.0, 3.0, 4.0]), np.array([1, 0, 1]), np.array([0, 2, 3])), shape=(2, 2))
    for matrix in (duplicated, unsorted):
        indices, data = matrix.indices.copy(), matrix.data.copy()
        solver = la.EigenSparseLU()
        assert solver.compute(matrix)
        rhs = np.array([1.0, 2.0])
        np.testing.assert_allclose(matrix.toarray() @ solver.solve(rhs), rhs, atol=1e-12)
        np.testing.assert_array_equal(matrix.indices, indices)
        np.testing.assert_array_equal(matrix.data, data)


def test_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        la.EigenLDLT().compute(sp.csc_matrix((3, 4)))


def test_solve_into_provided_array():
    A = laplacian(6)
    solver = la.EigenLDLT()
    assert solver.compute(A)
    x = np.zeros(6)
    solver.solve(np.ones(6), x)
    np.testing.assert_allclose(A @ x, np.ones(6), atol=1e-12)


@pytest.mark.parametrize("x", [np.zeros(12)[::2], np.zeros(6, dtype=np.float32), np.zeros(6).reshape(1, 6)],
                         ids=["strided", "float32", "2d"])
def test_solve_into_rejects_arrays_it_cannot_write(x):
    solver = la.EigenLDLT()
    assert solver.compute(laplacian(6))
    with pytest.raises(TypeError):
        solver.solve(np.ones(6), x)


def test_vectors_must_be_arrays():
    solver = la.EigenLDLT()
    assert solver.compute(laplacian(3))
    with pytest.raises(TypeError):
        solver.solve([1.0, 2.0, 3.0])


def test_solve_requires_factorization():
    with pytest.raises(RuntimeError, match="factorize"):
        la.EigenLDLT().solve(np.ones(3))


def test_solve_checks_size():
    solver = la.EigenLDLT()
    assert solver.compute(laplacian(4))
    with pytest.raises(ValueError):
        solver.solve(np.ones(5))


def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        la.SparseSolver()
    assert all(issubclass(solver_type, la.SparseSolver) for solver_type in ALL_SOLVERS)


def test_factorize_reuses_the_analysis_for_new_values():
    solver = la.EigenLDLT()
    assert solver.analyze(laplacian(20))
    for scale in (1.0, 3.5):
        A = laplacian(20, scale)
        assert solver.factorize(A)
        np.testing.assert_allclose(A @ solver.solve(np.ones(20)), np.ones(20), atol=1e-12)


def test_factorize_requires_an_analysis():
    with pytest.raises(RuntimeError, match="analyze"):
        la.EigenLDLT().factorize(laplacian(4))


def test_factorize_rejects_another_pattern():
    solver = la.EigenLDLT()
    assert solver.analyze(laplacian(4))
    # Same size and nnz as laplacian(4), different positions.
    lookalike = sp.csc_matrix(np.array([[4, 0, 1, 1], [0, 4, 0, 1], [1, 0, 4, 0], [1, 1, 0, 4]], dtype=float))
    assert lookalike.nnz == laplacian(4).nnz
    with pytest.raises(ValueError, match="analyzed sparsity pattern"):
        solver.factorize(lookalike)
    assert solver.factorize(laplacian(4))


def test_analyze_discards_the_factorization():
    solver = la.EigenLDLT()
    assert solver.compute(laplacian(4))
    assert solver.analyze(laplacian(4))
    with pytest.raises(RuntimeError):
        solver.solve(np.ones(4))


def test_auto_aliases_are_the_platform_classes():
    if HAS_ACCELERATE:
        assert la.AutoSparseSolver is la.AccelerateLLT
        assert la.AutoIndefiniteSparseSolver is la.AccelerateLDLT
    elif HAS_MKL:
        assert la.AutoSparseSolver is la.PardisoLLT
        assert la.AutoIndefiniteSparseSolver is la.PardisoLDLT
    else:
        assert la.AutoSparseSolver is la.EigenLDLT
        assert la.AutoIndefiniteSparseSolver is la.EigenSparseLU


@pytest.mark.skipif(sys.platform == "darwin", reason="Accelerate is built on macOS")
def test_accelerate_absent_elsewhere():
    with pytest.raises(ImportError):
        from litepgo.linalg.sparse_solver import AccelerateLLT  # noqa: F401


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="oneMKL is built on Linux")
def test_pardiso_absent_elsewhere():
    with pytest.raises(ImportError):
        from litepgo.linalg.sparse_solver import PardisoLLT  # noqa: F401


def test_parent_module_preserves_solver_exports():
    import litepgo.linalg as parent

    for name in la.__all__:
        assert getattr(parent, name) is getattr(la, name)
