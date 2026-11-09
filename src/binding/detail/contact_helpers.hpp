#pragma once
#include "litepgo/contact/ipc_contact_energy.hpp"
#include <memory>
#include <nanobind/nanobind.h>

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <type_traits>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
template <class ContactEnergy, class Options> void bind_backend(nb::module_& m, const char* energy_name) {
    auto cls = nb::class_<ContactEnergy, energy::Energy>(m, energy_name);
    if constexpr (std::is_same_v<ContactEnergy, IPCContactEnergy>) {
        cls.def(nb::init<const mesh::TriangleMesh&, Options, std::shared_ptr<const BroadPhase>,
                         std::optional<ContactPairWeights>>(),
                "surface"_a, "options"_a, "broad_phase"_a, "pair_weights"_a = nb::none(),
                "Native IPC self-contact barrier with a required broad phase.");
    } else {
        cls.def(nb::init<const mesh::TriangleMesh&, Options, std::optional<ContactPairWeights>>(), "surface"_a,
                "options"_a, "pair_weights"_a = nb::none(),
                "IPC Toolkit self-contact barrier over surface vertex positions.");
    }
    cls.def_prop_ro("num_vertices", &ContactEnergy::num_vertices)
        .def_prop_ro(
            "rest_positions",
            [](const ContactEnergy& energy) -> Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> {
                return energy.rest_positions().template reshaped<Eigen::RowMajor>(energy.num_vertices(), 3);
            },
            "The surface's rest positions, one row per vertex (a copy).")
        .def_prop_ro(
            "options", [](const ContactEnergy& energy) { return Options(energy.options()); },
            "A copy of the options used to construct this energy.")
        .def("num_active_pairs", &ContactEnergy::num_active_pairs, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("min_active_distance", &ContactEnergy::min_active_distance, "x"_a,
             nb::call_guard<nb::gil_scoped_release>(), "Smallest distance among the active pairs; inf if none.")
        .def("contact_snapshot", &ContactEnergy::contact_snapshot, "x"_a, nb::call_guard<nb::gil_scoped_release>(),
             "Copy all merged active records at the entire surface's flat xyz positions, including zero-mollifier EE. "
             "The snapshot is independent of future queries and does not update the Hessian pattern. "
             "Its record count equals num_active_pairs(x); this is not an intersection test.");
}
} // namespace litepgo::contact::binding
