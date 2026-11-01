#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/bindings/bindings.hpp"
#include "litepgo/bindings/detail/fem_helpers.hpp"
#include "litepgo/bindings/detail/templates.hpp"

#include <nanobind/eigen/dense.h>

#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace litepgo::fem::binding {
namespace {

template <int N> void bind_gauss_legendre_2d(nb::module_& m) {
    const std::string name = "GaussLegendre3D[" + std::to_string(N) + "]";
    bind_quadrature<GaussLegendre3D<N>>(m, name.c_str());
    register_instance(m, "GaussLegendre3D", nb::make_tuple(N), nb::type<GaussLegendre3D<N>>());
}
} // namespace

void bind_fem_quadrature_3d_gauss_legendre_3d(nb::module_& m) {
    bind_gauss_legendre_2d<2>(m);
    bind_gauss_legendre_2d<4>(m);
}
} // namespace litepgo::fem::binding
