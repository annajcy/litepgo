#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"

#include <string>

namespace nb = nanobind;

namespace litepgo::fem::binding {
namespace {
template <int N> void bind_gauss_legendre_1d(nb::module_& m) {
    const std::string name = "GaussLegendre1D[" + std::to_string(N) + "]";
    bind_quadrature<GaussLegendre1D<N>>(m, name.c_str());
    register_instance(m, "GaussLegendre1D", nb::make_tuple(N), nb::type<GaussLegendre1D<N>>());
}
} // namespace

void bind_fem_quadrature_1d_gauss_legendre_1d(nb::module_& m) {
    bind_gauss_legendre_1d<2>(m);
    bind_gauss_legendre_1d<4>(m);
}
} // namespace litepgo::fem::binding
