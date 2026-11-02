#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include "bindings.hpp"
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::mesh::binding {
namespace {
template <int K> using Cells = Eigen::Matrix<int, Eigen::Dynamic, K, Eigen::RowMajor>;
} // namespace

void bind_triangle_mid_edge_stencils(nb::module_& m) {
    nb::class_<TriangleMidEdgeStencils>(m, "TriangleMidEdgeStencils")
        .def_ro("vertices", &TriangleMidEdgeStencils::vertices,
                "Per face: its three vertices, then the unique opposite vertices across edges 0, 1, 2.")
        .def_prop_ro(
            "opposite",
            [](const TriangleMidEdgeStencils& stencils) {
                Cells<3> opposite(static_cast<Eigen::Index>(stencils.opposite.size()), 3);
                for (std::size_t f = 0; f < stencils.opposite.size(); ++f)
                    for (int k = 0; k < 3; ++k)
                        opposite(static_cast<Eigen::Index>(f), k) = stencils.opposite[f][static_cast<std::size_t>(k)];
                return opposite;
            },
            "Per face: the local row across each edge, or -1 at a boundary (a copy).");
    m.def("make_triangle_mid_edge_stencils", &make_triangle_mid_edge_stencils, "topology"_a);
}
} // namespace litepgo::mesh::binding
