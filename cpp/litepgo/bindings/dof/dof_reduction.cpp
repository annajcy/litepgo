#include "litepgo/dof/dof_reduction.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::dof::binding {
void bind_dof_reduction(nb::module_& m) {

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
            "reduce",
            [](const DofReduction& reduction, const linalg::binding::SparseArgument& A) {
                return reduction.reduce(A.matrix);
            },
            "A"_a, "Principal submatrix of a scipy.sparse matrix on the free DOFs.")
        .def("expand", &DofReduction::expand, "reduced"_a, "full"_a.noconvert(),
             "Write reduced into the free entries of full, a contiguous writable float64 array.");
}
} // namespace litepgo::dof::binding
