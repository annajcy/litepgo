#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"
#include "bindings.hpp"
#include "detail/sparse_matrix.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {
void bind_sparse_solver(nb::module_& m) {

    // Matrices arrive as scipy.sparse and are copied into C++ (see
    // SparseArgument). The GIL is released while factoring and solving.
    nb::class_<SparseSolver>(m, "SparseSolver")
        .def(
            "analyze", [](SparseSolver& solver, const SparseArgument& A) { return solver.analyze(A.matrix); }, "A"_a,
            nb::call_guard<nb::gil_scoped_release>(),
            "Symbolic analysis of A's sparsity pattern; returns False if it cannot be analyzed.")
        .def(
            "factorize", [](SparseSolver& solver, const SparseArgument& A) { return solver.factorize(A.matrix); },
            "A"_a, nb::call_guard<nb::gil_scoped_release>(),
            "Numeric factorization of A, which must have the analyzed pattern; returns False on numerical "
            "failure.")
        .def(
            "compute", [](SparseSolver& solver, const SparseArgument& A) { return solver.compute(A.matrix); }, "A"_a,
            nb::call_guard<nb::gil_scoped_release>(), "analyze(A), then factorize(A).")
        .def("solve", nb::overload_cast<const ConstVecRef&>(&SparseSolver::solve, nb::const_), "rhs"_a,
             nb::call_guard<nb::gil_scoped_release>(), "Solve with the last successful factorization.")
        .def("solve", nb::overload_cast<const ConstVecRef&, VecRef>(&SparseSolver::solve, nb::const_), "rhs"_a,
             "x"_a.noconvert(), nb::call_guard<nb::gil_scoped_release>(),
             "Solve into x, a contiguous writable float64 array.");
}
} // namespace litepgo::linalg::binding
