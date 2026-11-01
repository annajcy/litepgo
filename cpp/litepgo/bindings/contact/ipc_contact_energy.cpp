#include "litepgo/contact/ipc_contact_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/contact_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
void bind_ipc_contact_energy(nb::module_& m) {
    bind_backend<IPCContactEnergy, IPCContactOptions>(m, "IPCContactEnergy");
}
} // namespace litepgo::contact::binding
