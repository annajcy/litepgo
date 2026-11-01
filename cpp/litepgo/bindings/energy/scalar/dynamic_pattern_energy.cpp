#include "litepgo/energy/scalar/dynamic_pattern_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/energy_helpers.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include <nanobind/trampoline.h>

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
namespace {
class PyDynamicEnergy : public PyEnergy<DynamicPatternEnergy> {
  public:
    using PyEnergy<DynamicPatternEnergy>::PyEnergy;

  private:
    SpMat pattern_at(const ConstVecRef& x) const override {
        nb::detail::ticket ticket(this->nb_trampoline, "pattern_at", nb::detail::str_hash("pattern_at"), true);
        return nb::cast<linalg::binding::SparseArgument>(this->nb_trampoline.base().attr(ticket.key)(nb::cast(Vec(x))))
            .matrix;
    }
};
} // namespace

void bind_dynamic_pattern_energy(nb::module_& m) {
    nb::class_<DynamicPatternEnergy, Energy, PyDynamicEnergy>(m, "DynamicPatternEnergy")
        .def(nb::init<int, const linalg::binding::SparseArgument&>(), "num_dofs"_a, "initial_pattern"_a)
        .def(
            "feasible_step", [](const DynamicPatternEnergy&, const ConstVecRef&, const ConstVecRef&) { return 1.0; },
            "x"_a, "dx"_a);
}
} // namespace litepgo::energy::binding
