#include "litepgo/bindings/bindings.hpp"
#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/sparse_solver/accelerate_llt.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::linalg::binding {
void bind_accelerate_llt([[maybe_unused]] nb::module_& m) {
#if defined(LITEPGO_HAS_ACCELERATE)
    nb::class_<AccelerateLLT, SparseSolver>(m, "AccelerateLLT").def(nb::init<>());
#endif
}
} // namespace litepgo::linalg::binding
