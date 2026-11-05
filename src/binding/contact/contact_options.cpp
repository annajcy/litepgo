#include "litepgo/contact/contact_options.hpp"
#include "bindings.hpp"
#include "litepgo/contact/contact_snapshot.hpp"
#include "litepgo/contact/plane_penalty_energy.hpp"

#include <limits>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::contact::binding {
namespace {
template <class Matrix> auto readonly_view(const Matrix& data) {
    // A non-null pointer keeps nanobind's empty-array fallback from dropping constness.
    static const typename Matrix::Scalar empty = 0;
    return Eigen::Map<const Matrix>(data.size() ? data.data() : &empty, data.rows(), data.cols());
}

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

    nb::class_<ContactSnapshot>(m, "ContactSnapshot",
                                "Owned copy of all merged active records and current surface positions. "
                                "Arrays are read-only and keep the snapshot alive. Row indices are not persistent IDs.")
        .def_prop_ro(
            "positions", [](const ContactSnapshot& s) { return readonly_view(s.positions); },
            nb::rv_policy::reference_internal,
            "All current surface positions, shape (N, 3), in original vertex numbering.")
        .def_prop_ro(
            "types", [](const ContactSnapshot& s) { return nb::tuple(nb::cast(s.types)); },
            "Tuple of M record types: VV, VE, VF or EE. Mollified EE always retains four support vertices.")
        .def_prop_ro(
            "vertex_ids", [](const ContactSnapshot& s) { return readonly_view(s.vertex_ids); },
            nb::rv_policy::reference_internal,
            "Shape (M, 4): VV vertices; VE vertex then edge; VF vertex then triangle; EE two edges. Pad with -1.")
        .def_prop_ro(
            "distances", [](const ContactSnapshot& s) { return readonly_view(s.distances); },
            nb::rv_policy::reference_internal, "M unsquared distances d, without subtracting min_distance.")
        .def_prop_ro(
            "weights", [](const ContactSnapshot& s) { return readonly_view(s.weights); },
            nb::rv_policy::reference_internal,
            "M merged candidate weights: reference areas when enabled, else multiplicities. Excludes kappa and "
            "mollifiers.")
        .def_prop_ro(
            "mollifiers", [](const ContactSnapshot& s) { return readonly_view(s.mollifiers); },
            nb::rv_policy::reference_internal, "M EE mollifiers, including zero; one for other record types.");

    nb::class_<PlanePenaltyEnergy, energy::Energy>(m, "PlanePenaltyEnergy")
        .def(nb::init<int, int, double, double, bool>(), "num_vertices"_a, "axis"_a, "height"_a, "stiffness"_a,
             "positive_side"_a = true, "One-sided quadratic penalty on vertex-major xyz positions.");

    bind_contact_options<IPCContactOptions>(m, "IPCContactOptions");
#ifdef LITEPGO_HAS_IPC_TOOLKIT
    bind_contact_options<IPCTKContactOptions>(m, "IPCTKContactOptions");
#endif
}
} // namespace litepgo::contact::binding
