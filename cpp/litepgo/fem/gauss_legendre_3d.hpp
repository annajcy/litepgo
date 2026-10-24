#pragma once

#include "litepgo/fem/gauss_legendre_1d.hpp"
#include "litepgo/fem/tensor_product_quadrature.hpp"

namespace litepgo {

template <int NumPoints> using GaussLegendre3D = TensorProductQuadrature3D<GaussLegendre1D<NumPoints>>;

} // namespace litepgo
