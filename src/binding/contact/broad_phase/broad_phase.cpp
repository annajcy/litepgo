#include "litepgo/contact/broad_phase/broad_phase.hpp"
#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
void bind_broad_phase(nb::module_& m) {
    nb::class_<BroadPhase>(m, "BroadPhase");
}
} // namespace litepgo::contact::binding
