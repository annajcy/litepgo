#include "litepgo/fem/embedding/point_displacement_map.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"
#include <nanobind/eigen/sparse.h>

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <class B> void bind_point_displacement_map(nb::module_& m, const char* name) {
    m.def(
        name,
        [](const PointEmbedding& embedding, const dof::VertexDofMap& map, const Responses& fields) {
            return build_point_displacement_map<B>(embedding, map, fields);
        },
        "embedding"_a, "u_dof_map"_a, "X_fields"_a);
    register_instance(m, "build_point_displacement_map", nb::make_tuple(nb::type<B>()), m.attr(name));
}
} // namespace

void bind_fem_point_displacement_map(nb::module_& m) {
    bind_point_displacement_map<TetP1Basis>(m, "_point_map_tet_p1");
    bind_point_displacement_map<HexQ1Basis>(m, "_point_map_hex_q1");
    bind_point_displacement_map<HexHermiteBasis>(m, "_point_map_hex_hermite");
}
} // namespace litepgo::fem::binding
