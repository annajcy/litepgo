#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"
#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"

namespace litepgo::fem::binding {
void bind_basis_1d(nanobind::module_& m) {
    bind_basis<LinearLagrangeBasis, 1>(m, "LinearLagrangeBasis");
    bind_basis<CubicHermiteBasis, 1>(m, "CubicHermiteBasis");
}
} // namespace litepgo::fem::binding
