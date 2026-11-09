#include "litepgo/contact/contact_options.hpp"
#include "bindings.hpp"
#include "litepgo/contact/contact_pair_weights.hpp"
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
               bool use_area_weighting, bool use_physical_barrier) {
                new (self) Options{
                    dhat, min_distance, ccd_tolerance, ccd_max_iterations, use_area_weighting, use_physical_barrier};
            },
            "dhat"_a = std::numeric_limits<double>::quiet_NaN(), "min_distance"_a = 0.0,
            "ccd_tolerance"_a = Options{}.ccd_tolerance, "ccd_max_iterations"_a = Options{}.ccd_max_iterations,
            "use_area_weighting"_a = Options{}.use_area_weighting,
            "use_physical_barrier"_a = Options{}.use_physical_barrier)
        .def_rw("dhat", &Options::dhat, "Physical activation width above min_distance, matching IPC Toolkit.")
        .def_rw("min_distance", &Options::min_distance, "Minimum physical separation, matching IPC Toolkit's dmin.")
        .def_rw("ccd_tolerance", &Options::ccd_tolerance)
        .def_rw("ccd_max_iterations", &Options::ccd_max_iterations)
        .def_rw("use_area_weighting", &Options::use_area_weighting)
        .def_rw("use_physical_barrier", &Options::use_physical_barrier,
                "IPC Toolkit physical normalization. With area weights the external stiffness has pressure units.");
}
} // namespace

void bind_contact_contact_options(nb::module_& m) {

    nb::class_<ContactPairWeights>(m, "ContactPairWeights",
                                   "Fixed primitive groups and multipliers. Inputs and exposed properties are copies; "
                                   "energies own their configuration.")
        .def(
            "__init__",
            [](ContactPairWeights* self, std::vector<int> vertices, std::vector<int> edges, std::vector<int> faces,
               nb::object weights) {
                const auto numpy = nb::module_::import_("numpy");
                if (nb::cast<bool>(numpy.attr("iscomplexobj")(weights)))
                    throw std::invalid_argument("ContactPairWeights requires real multipliers");
                const auto array = numpy.attr("asarray")(weights, "dtype"_a = "float64");
                if (nb::cast<int>(array.attr("ndim")) != 2)
                    throw std::invalid_argument("ContactPairWeights requires a two-dimensional multiplier table");
                auto matrix = nb::cast<Eigen::MatrixXd>(array);
                new (self)
                    ContactPairWeights{std::move(vertices), std::move(edges), std::move(faces), std::move(matrix)};
            },
            "vertex_groups"_a, "edge_groups"_a, "face_groups"_a, "weights"_a)
        .def_prop_ro(
            "vertex_groups", [](const ContactPairWeights& p) { return p.vertex_groups; },
            "One integer group per input vertex (a copy).")
        .def_prop_ro(
            "edge_groups", [](const ContactPairWeights& p) { return p.edge_groups; },
            "One group per contact_edges(topology) row (a copy).")
        .def_prop_ro(
            "face_groups", [](const ContactPairWeights& p) { return p.face_groups; },
            "One group per input triangle (a copy).")
        .def_prop_ro(
            "weights", [](const ContactPairWeights& p) { return Eigen::MatrixXd(p.weights); },
            "Symmetric finite nonnegative G x G table (a copy); validated when an energy is constructed.");
    m.def(
        "contact_edges",
        [](const mesh::TriangleTopology& topology) {
            const auto edges = contact_edges(topology);
            Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor> result(edges.size(), 2);
            for (std::size_t i = 0; i < edges.size(); ++i) {
                result(i, 0) = edges[i][0];
                result(i, 1) = edges[i][1];
            }
            return result;
        },
        "topology"_a,
        "Unique undirected edges (min vertex, max vertex), sorted lexicographically; edge_groups uses this order.");

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
            "M sums of candidate area weights (or one) times optional group multipliers. Excludes kappa, "
            "barrier normalization and mollifiers.")
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
