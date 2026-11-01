#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/energy_helpers.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"
#include "litepgo/energy/scalar/energy.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_fixed_pattern_energy(nb::module_& m) {
    nb::class_<FixedPatternEnergy, Energy, PyEnergy<FixedPatternEnergy>>(m, "FixedPatternEnergy")
        .def(nb::init<int, const linalg::binding::SparseArgument&>(), "num_dofs"_a, "pattern"_a)
        .def(
            "feasible_step", [](const FixedPatternEnergy&, const ConstVecRef&, const ConstVecRef&) { return 1.0; },
            "x"_a, "dx"_a);
}
} // namespace litepgo::energy::binding
