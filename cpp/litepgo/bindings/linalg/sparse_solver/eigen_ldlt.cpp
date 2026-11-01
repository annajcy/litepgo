#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {
void bind_eigen_ldlt(nb::module_& m) {
    nb::class_<EigenLDLT, SparseSolver>(m, "EigenLDLT").def(nb::init<>());
}
} // namespace litepgo::linalg::binding
