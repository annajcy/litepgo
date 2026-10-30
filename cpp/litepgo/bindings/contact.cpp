#include "litepgo/bindings/bindings.hpp"
#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/broad_phase/sweep_and_prune.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"
#ifdef LITEPGO_HAS_IPC_TOOLKIT
#include "litepgo/contact/ipctk_contact_energy.hpp"
#endif

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <type_traits>

#include <limits>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {

template <class ContactEnergy, class Options>
void bind_backend(nb::module_& m, const char* energy_name, const char* options_name) {
    nb::class_<Options>(m, options_name)
        .def(
            "__init__",
            [](Options* self, double dhat, double min_distance, double ccd_tolerance, long ccd_max_iterations) {
                new (self) Options{dhat, min_distance, ccd_tolerance, ccd_max_iterations};
            },
            "dhat"_a = std::numeric_limits<double>::quiet_NaN(), "min_distance"_a = 0.0,
            "ccd_tolerance"_a = Options{}.ccd_tolerance, "ccd_max_iterations"_a = Options{}.ccd_max_iterations)
        .def_rw("dhat", &Options::dhat)
        .def_rw("min_distance", &Options::min_distance)
        .def_rw("ccd_tolerance", &Options::ccd_tolerance)
        .def_rw("ccd_max_iterations", &Options::ccd_max_iterations);

    auto cls = nb::class_<ContactEnergy, energy::Energy>(m, energy_name);
    if constexpr (std::is_same_v<ContactEnergy, IPCContactEnergy>) {
        cls.def(nb::init<const mesh::TriangleMesh&, Options, std::shared_ptr<const BroadPhase>, std::vector<Plane>>(),
                "surface"_a, "options"_a, "broad_phase"_a, "planes"_a = std::vector<Plane>(),
                "Native IPC barrier; a non-null broad phase is required and retained.");
    } else {
        cls.def(nb::init<const mesh::TriangleMesh&, Options, std::vector<Plane>>(), "surface"_a, "options"_a,
                "planes"_a = std::vector<Plane>(),
                "IPC barrier of a triangle surface and static planes, over the surface vertices' positions.");
    }
    cls.def_prop_ro("num_vertices", &ContactEnergy::num_vertices)
        .def_prop_ro(
            "rest_positions",
            [](const ContactEnergy& energy) -> Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> {
                return energy.rest_positions().template reshaped<Eigen::RowMajor>(energy.num_vertices(), 3);
            },
            "The surface's rest positions, one row per vertex (a copy).")
        .def_prop_ro("options", &ContactEnergy::options)
        .def("num_active_pairs", &ContactEnergy::num_active_pairs, "x"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("min_active_distance", &ContactEnergy::min_active_distance, "x"_a,
             nb::call_guard<nb::gil_scoped_release>(), "Smallest distance among the active pairs; inf if none.");
}

void bind(nb::module_& m) {
    m.doc() = "Native IPC and optional IPC Toolkit contact (cpp/litepgo/contact)";

    nb::class_<Plane>(m, "Plane")
        .def(
            "__init__",
            [](Plane* self, const Eigen::Vector3d& point, const Eigen::Vector3d& normal) {
                new (self) Plane{point, normal};
            },
            "point"_a, "normal"_a, "A static plane; vertices stay on the side the normal points to.")
        .def_rw("point", &Plane::point)
        .def_rw("normal", &Plane::normal);

    nb::class_<BroadPhase>(m, "BroadPhase");
    nb::class_<SweepAndPrune, BroadPhase>(m, "SweepAndPrune").def(nb::init<>());

    nb::class_<SpatialHash, BroadPhase>(m, "SpatialHash")
        .def(nb::init<std::optional<double>>(), "cell_size"_a = nb::none());

    bind_backend<IPCContactEnergy, IPCContactOptions>(m, "IPCContactEnergy", "IPCContactOptions");
#ifdef LITEPGO_HAS_IPC_TOOLKIT
    bind_backend<IPCTKContactEnergy, IPCTKContactOptions>(m, "IPCTKContactEnergy", "IPCTKContactOptions");
#endif
}

} // namespace litepgo::contact::binding
