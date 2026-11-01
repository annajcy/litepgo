#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_quad_hermite_basis(nb::module_& m) {
    bind_basis<QuadHermiteBasis, 2>(m, "QuadHermiteBasis");
}
} // namespace litepgo::fem::binding
