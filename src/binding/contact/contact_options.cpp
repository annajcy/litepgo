#include "litepgo/contact/contact_options.hpp"
#include "bindings.hpp"
#include "litepgo/contact/plane_penalty_energy.hpp"

#include <limits>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
namespace {
template <class Options> void bind_contact_options(nb::module_& m, const char* options_name) {
    nb::class_<Options>(m, options_name)
        .def(
            "__init__",
            [](Options* self, double dhat, double min_distance, double ccd_tolerance, long ccd_max_iterations,
               bool use_area_weighting) {
                new (self) Options{dhat, min_distance, ccd_tolerance, ccd_max_iterations, use_area_weighting};
            },
            "dhat"_a = std::numeric_limits<double>::quiet_NaN(), "min_distance"_a = 0.0,
            "ccd_tolerance"_a = Options{}.ccd_tolerance, "ccd_max_iterations"_a = Options{}.ccd_max_iterations,
            "use_area_weighting"_a = Options{}.use_area_weighting)
        .def_rw("dhat", &Options::dhat)
        .def_rw("min_distance", &Options::min_distance)
        .def_rw("ccd_tolerance", &Options::ccd_tolerance)
        .def_rw("ccd_max_iterations", &Options::ccd_max_iterations)
        .def_rw("use_area_weighting", &Options::use_area_weighting);
}
} // namespace

void bind_contact_contact_options(nb::module_& m) {

    nb::class_<PlanePenaltyEnergy, energy::Energy>(m, "PlanePenaltyEnergy")
        .def(nb::init<int, int, double, double, bool>(), "num_vertices"_a, "axis"_a, "height"_a, "stiffness"_a,
             "positive_side"_a = true, "One-sided quadratic penalty on vertex-major xyz positions.");

    bind_contact_options<IPCContactOptions>(m, "IPCContactOptions");
#ifdef LITEPGO_HAS_IPC_TOOLKIT
    bind_contact_options<IPCTKContactOptions>(m, "IPCTKContactOptions");
#endif
}
} // namespace litepgo::contact::binding
