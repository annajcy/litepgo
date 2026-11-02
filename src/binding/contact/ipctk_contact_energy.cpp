#include "bindings.hpp"
#include "detail/contact_helpers.hpp"
#ifdef LITEPGO_HAS_IPC_TOOLKIT
#include "litepgo/contact/ipctk_contact_energy.hpp"
#endif

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
void bind_ipctk_contact_energy([[maybe_unused]] nb::module_& m) {
#ifdef LITEPGO_HAS_IPC_TOOLKIT
    bind_backend<IPCTKContactEnergy, IPCTKContactOptions>(m, "IPCTKContactEnergy");
#endif
}
} // namespace litepgo::contact::binding
