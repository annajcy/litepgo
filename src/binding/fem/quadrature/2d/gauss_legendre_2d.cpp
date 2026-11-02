#include "litepgo/fem/quadrature/2d/gauss_legendre_2d.hpp"
#include "bindings.hpp"
#include "detail/fem_helpers.hpp"
#include "detail/templates.hpp"

#include <nanobind/eigen/dense.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <int N> void bind_gauss_legendre_2d(nb::module_& m) {
    const std::string name = "GaussLegendre2D[" + std::to_string(N) + "]";
    bind_quadrature<GaussLegendre2D<N>>(m, name.c_str());
    register_instance(m, "GaussLegendre2D", nb::make_tuple(N), nb::type<GaussLegendre2D<N>>());
}
} // namespace

void bind_fem_quadrature_2d_gauss_legendre_2d(nb::module_& m) {
    bind_gauss_legendre_2d<2>(m);
    bind_gauss_legendre_2d<4>(m);
}
} // namespace litepgo::fem::binding
