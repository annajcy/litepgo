#include "litepgo/linalg/sparse_solver/eigen_sparse_lu.hpp"
#include "bindings.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {
void bind_eigen_sparse_lu(nb::module_& m) {
    nb::class_<EigenSparseLU, SparseSolver>(m, "EigenSparseLU").def(nb::init<>());
}
} // namespace litepgo::linalg::binding
