#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_triangle_p1_basis(nb::module_& m) {
    bind_basis<TriangleP1Basis, 2>(m, "TriangleP1Basis");
}
} // namespace litepgo::fem::binding
