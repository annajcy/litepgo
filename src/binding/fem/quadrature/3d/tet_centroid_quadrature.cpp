#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_tet_centroid_quadrature(nb::module_& m) {
    bind_quadrature<TetCentroidQuadrature>(m, "TetCentroidQuadrature");
}
} // namespace litepgo::fem::binding
