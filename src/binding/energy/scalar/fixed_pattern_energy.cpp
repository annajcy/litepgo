#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"
#include "bindings.hpp"
#include "detail/energy_helpers.hpp"
#include "detail/sparse_matrix.hpp"
#include "litepgo/energy/scalar/energy.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_fixed_pattern_energy(nb::module_& m) {
    nb::class_<FixedPatternEnergy, Energy, PyEnergy<FixedPatternEnergy>>(m, "FixedPatternEnergy")
        .def(nb::init<const linalg::binding::SparseArgument&>(), "pattern"_a)
        .def(
            "feasible_step", [](const FixedPatternEnergy&, const ConstVecRef&, const ConstVecRef&) { return 1.0; },
            "x"_a, "dx"_a);
}
} // namespace litepgo::energy::binding
