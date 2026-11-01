#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_hex_q1_basis(nb::module_& m) {
    bind_basis<HexQ1Basis, 3>(m, "HexQ1Basis");
}
} // namespace litepgo::fem::binding
