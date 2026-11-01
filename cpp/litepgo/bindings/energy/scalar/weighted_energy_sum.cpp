#include "litepgo/energy/scalar/weighted_energy_sum.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/energy/scalar/energy.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_weighted_energy_sum(nb::module_& m) {
    nb::class_<WeightedEnergySum, Energy>(m, "WeightedEnergySum")
        .def(
            "__init__",
            [](WeightedEnergySum* self, const std::vector<std::pair<std::shared_ptr<Energy>, double>>& terms) {
                std::vector<WeightedEnergy> weighted;
                for (const auto& [energy, weight] : terms) {
                    weighted.push_back({energy, weight});
                }
                new (self) WeightedEnergySum(std::move(weighted));
            },
            "terms"_a, "sum_i w_i E_i(x) from (energy, weight) pairs over the same DOFs; weights are nonzero.");
}
} // namespace litepgo::energy::binding
