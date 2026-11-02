#include "litepgo/contact/contact_options.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>

#include <limits>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
namespace {
template <class Options> void bind_contact_options(nb::module_& m, const char* options_name) {
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
}
} // namespace

void bind_contact_contact_options(nb::module_& m) {

    nb::class_<Plane>(m, "Plane")
        .def(
            "__init__",
            [](Plane* self, const Eigen::Vector3d& point, const Eigen::Vector3d& normal) {
                new (self) Plane{point, normal};
            },
            "point"_a, "normal"_a, "A static plane; vertices stay on the side the normal points to.")
        .def_rw("point", &Plane::point)
        .def_rw("normal", &Plane::normal);

    bind_contact_options<IPCContactOptions>(m, "IPCContactOptions");
#ifdef LITEPGO_HAS_IPC_TOOLKIT
    bind_contact_options<IPCTKContactOptions>(m, "IPCTKContactOptions");
#endif
}
} // namespace litepgo::contact::binding
