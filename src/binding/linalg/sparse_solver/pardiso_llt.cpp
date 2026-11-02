#include "bindings.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#if defined(LITEPGO_HAS_MKL)
#include "litepgo/linalg/sparse_solver/pardiso_llt.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {
void bind_pardiso_llt([[maybe_unused]] nb::module_& m) {
#if defined(LITEPGO_HAS_MKL)
    nb::class_<PardisoLLT, SparseSolver>(m, "PardisoLLT").def(nb::init<>());
#endif
}
} // namespace litepgo::linalg::binding
