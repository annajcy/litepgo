#include "bindings.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#if defined(LITEPGO_HAS_MKL)
#include "litepgo/linalg/sparse_solver/pardiso_ldlt.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {
void bind_pardiso_ldlt([[maybe_unused]] nb::module_& m) {
#if defined(LITEPGO_HAS_MKL)
    nb::class_<PardisoLDLT, SparseSolver>(m, "PardisoLDLT").def(nb::init<>());
#endif
}
} // namespace litepgo::linalg::binding
