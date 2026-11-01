#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"

#include <nanobind/eigen/dense.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
void bind_tet_p1_basis(nb::module_& m) {
    bind_basis<TetP1Basis, 3>(m, "TetP1Basis");
}
} // namespace litepgo::fem::binding
