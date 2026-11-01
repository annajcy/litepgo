#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {
void bind_auto_sparse_solver(nb::module_& m) {
    // The C++ aliases: the same class objects under a second name.
    m.attr("AutoSparseSolver") = nb::type<AutoSparseSolver>();
    m.attr("AutoIndefiniteSparseSolver") = nb::type<AutoIndefiniteSparseSolver>();
}
} // namespace litepgo::linalg::binding
