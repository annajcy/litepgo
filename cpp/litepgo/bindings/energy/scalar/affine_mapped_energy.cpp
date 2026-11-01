#include "litepgo/energy/scalar/affine_mapped_energy.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/sparse_matrix.hpp"
#include "litepgo/energy/scalar/energy.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <memory>
#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::energy::binding {
void bind_affine_mapped_energy(nb::module_& m) {
    nb::class_<AffineMappedEnergy, Energy>(m, "AffineMappedEnergy")
        .def(
            "__init__",
            [](AffineMappedEnergy* self, std::shared_ptr<Energy> energy, const linalg::binding::SparseArgument& T,
               std::optional<Vec> offset) {
                new (self) AffineMappedEnergy(std::move(energy), T.matrix, offset.value_or(Vec()));
            },
            "energy"_a, "T"_a, "offset"_a = nb::none(), "E(z) = energy(T z + offset).");
}
} // namespace litepgo::energy::binding
