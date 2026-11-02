#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "bindings.hpp"
#include <optional>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
void bind_spatial_hash(nb::module_& m) {
    nb::class_<SpatialHash, BroadPhase>(m, "SpatialHash")
        .def(nb::init<std::optional<double>>(), "cell_size"_a = nb::none());
}
} // namespace litepgo::contact::binding
