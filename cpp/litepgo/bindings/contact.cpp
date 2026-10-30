#include "litepgo/bindings/bindings.hpp"
#include "litepgo/contact/barrier_contact_energy.hpp"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

#include <limits>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {

void bind(nb::module_& m) {
    m.doc() = "IPC barrier contact on IPC Toolkit (cpp/litepgo/contact)";

    nb::class_<Plane>(m, "Plane")
        .def(
            "__init__",
            [](Plane* self, const Eigen::Vector3d& point, const Eigen::Vector3d& normal) {
                new (self) Plane{point, normal};
            },
            "point"_a, "normal"_a, "A static plane; vertices stay on the side the normal points to.")
        .def_rw("point", &Plane::point)
        .def_rw("normal", &Plane::normal);

    nb::class_<BarrierContactOptions>(m, "BarrierContactOptions")
        .def(
            "__init__",
            [](BarrierContactOptions* self, double dhat, double min_distance, double ccd_tolerance,
               long ccd_max_iterations) {
                new (self) BarrierContactOptions{dhat, min_distance, ccd_tolerance, ccd_max_iterations};
            },
            "dhat"_a = std::numeric_limits<double>::quiet_NaN(), "min_distance"_a = 0.0,
            "ccd_tolerance"_a = BarrierContactOptions{}.ccd_tolerance,
            "ccd_max_iterations"_a = BarrierContactOptions{}.ccd_max_iterations)
        .def_rw("dhat", &BarrierContactOptions::dhat)
        .def_rw("min_distance", &BarrierContactOptions::min_distance)
        .def_rw("ccd_tolerance", &BarrierContactOptions::ccd_tolerance)
        .def_rw("ccd_max_iterations", &BarrierContactOptions::ccd_max_iterations);

    nb::class_<BarrierContactEnergy, energy::Energy>(m, "BarrierContactEnergy")
        .def(nb::init<const mesh::TriangleMesh&, BarrierContactOptions, std::vector<Plane>>(), "surface"_a, "options"_a,
             "planes"_a = std::vector<Plane>(),
             "IPC barrier of a triangle surface and static planes, over the surface vertices' positions.")
        .def_prop_ro("num_vertices", &BarrierContactEnergy::num_vertices)
        .def_prop_ro(
            "rest_positions",
            [](const BarrierContactEnergy& energy) -> Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> {
                return energy.rest_positions().reshaped<Eigen::RowMajor>(energy.num_vertices(), 3);
            },
            "The surface's rest positions, one row per vertex (a copy).")
        .def_prop_ro("options", &BarrierContactEnergy::options)
        .def("num_active_pairs", &BarrierContactEnergy::num_active_pairs, "x"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("min_active_distance", &BarrierContactEnergy::min_active_distance, "x"_a,
             nb::call_guard<nb::gil_scoped_release>(), "Smallest distance among the active pairs; inf if none.");
}

} // namespace litepgo::contact::binding
