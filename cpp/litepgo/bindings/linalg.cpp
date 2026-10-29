#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/sparse_matrix.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/eigen_sparse_lu.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/sparse_solver/accelerate_ldlt.hpp"
#include "litepgo/linalg/sparse_solver/accelerate_llt.hpp"
#endif

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {

void bind(nb::module_& m) {
    m.doc() = "Sparse linear solvers (cpp/litepgo/linalg)";

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

    nb::class_<EigenLDLT, SparseSolver>(m, "EigenLDLT").def(nb::init<>());
    nb::class_<EigenSparseLU, SparseSolver>(m, "EigenSparseLU").def(nb::init<>());
#if defined(LITEPGO_HAS_ACCELERATE)
    nb::class_<AccelerateLLT, SparseSolver>(m, "AccelerateLLT").def(nb::init<>());
    nb::class_<AccelerateLDLT, SparseSolver>(m, "AccelerateLDLT").def(nb::init<>());
#endif

    // The C++ aliases: the same class objects under a second name.
    m.attr("AutoSparseSolver") = nb::type<AutoSparseSolver>();
    m.attr("AutoIndefiniteSparseSolver") = nb::type<AutoIndefiniteSparseSolver>();
}

} // namespace litepgo::linalg::binding
