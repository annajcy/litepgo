#include "litepgo/fem/quadrature/2d/triangle_centroid_quadrature.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_triangle_centroid_quadrature(nb::module_& m) {
    bind_quadrature<TriangleCentroidQuadrature>(m, "TriangleCentroidQuadrature");
}
} // namespace litepgo::fem::binding
