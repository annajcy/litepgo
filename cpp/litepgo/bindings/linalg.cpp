#include "litepgo/bindings/bindings.h"
#include "litepgo/bindings/sparse_matrix.h"
#include "litepgo/linalg/auto_sparse_solver.h"
#include "litepgo/linalg/dof_reduction.h"
#include "litepgo/linalg/eigen_ldlt.h"
#include "litepgo/linalg/eigen_sparse_lu.h"
#include "litepgo/linalg/sparse_solver.h"

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/accelerate_ldlt.h"
#include "litepgo/linalg/accelerate_llt.h"
#endif

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::bindings {

void bind_linalg(nb::module_& m) {
    m.doc() = "Sparse linear solvers (cpp/litepgo/linalg)";

    // Matrices arrive as scipy.sparse and are copied into C++ (see
    // SparseArgument). The GIL is released while factoring and solving.
    nb::class_<SparseSolver>(m, "SparseSolver")
        .def(
            "factorize", [](SparseSolver& solver, const SparseArgument& A) { return solver.factorize(A.matrix); },
            "A"_a, nb::call_guard<nb::gil_scoped_release>(),
            "Factor A, a scipy.sparse matrix; returns False on numerical failure.")
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

    nb::class_<DofReduction>(m, "DofReduction")
        .def(nb::init<int, std::vector<int>>(), "num_dofs"_a, "fixed_dofs"_a)
        .def_prop_ro("num_dofs", &DofReduction::num_dofs)
        .def_prop_ro("num_free", &DofReduction::num_free)
        .def_prop_ro(
            "fixed_dofs",
            [](const DofReduction& reduction) {
                const auto dofs = reduction.fixed_dofs();
                return Eigen::VectorXi(Eigen::Map<const Eigen::VectorXi>(dofs.data(), std::ssize(dofs)));
            },
            "Fixed DOFs in increasing order (a copy).")
        .def_prop_ro(
            "free_dofs",
            [](const DofReduction& reduction) {
                const auto dofs = reduction.free_dofs();
                return Eigen::VectorXi(Eigen::Map<const Eigen::VectorXi>(dofs.data(), std::ssize(dofs)));
            },
            "Free DOFs in increasing order (a copy).")
        .def("free_index", &DofReduction::free_index, "dof"_a)
        .def("reduce", nb::overload_cast<const ConstVecRef&>(&DofReduction::reduce, nb::const_), "full"_a,
             "Free entries of a vector.")
        .def(
            "reduce", [](const DofReduction& reduction, const SparseArgument& A) { return reduction.reduce(A.matrix); },
            "A"_a, "Principal submatrix of a scipy.sparse matrix on the free DOFs.")
        .def("expand", &DofReduction::expand, "reduced"_a, "full"_a.noconvert(),
             "Write reduced into the free entries of full, a contiguous writable float64 array.");

    // The C++ aliases: the same class objects under a second name.
    m.attr("AutoSparseSolver") = nb::type<AutoSparseSolver>();
    m.attr("AutoIndefiniteSparseSolver") = nb::type<AutoIndefiniteSparseSolver>();
}

} // namespace litepgo::bindings
