#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_hex_hermite_basis(nb::module_& m) {
    bind_basis<HexHermiteBasis, 3>(m, "HexHermiteBasis");
}
} // namespace litepgo::fem::binding
