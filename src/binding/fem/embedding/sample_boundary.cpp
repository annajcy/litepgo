#include "litepgo/fem/embedding/sample_boundary.hpp"
#include "bindings.hpp"
#include "detail/templates.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {
using bindings::register_instance;
} // namespace

void bind_sample_boundary(nb::module_& m) {
    nb::class_<SampledBoundary>(m, "SampledBoundary")
        .def_ro("mesh", &SampledBoundary::mesh)
        .def_ro("embedding", &SampledBoundary::embedding);
    m.def("_sample_boundary_hex_q1", &sample_boundary<HexQ1Basis>, "topology"_a, "X_fields"_a, "subdivisions"_a);
    register_instance(m, "sample_boundary", nb::make_tuple(nb::type<HexQ1Basis>()), m.attr("_sample_boundary_hex_q1"));
}
} // namespace litepgo::fem::binding
