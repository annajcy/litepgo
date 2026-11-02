#pragma once

#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/2d/tensor_product_quadrature_2d.hpp"

namespace litepgo::fem {

template <int PointsPerAxis> using GaussLegendre2D = TensorProductQuadrature2D<GaussLegendre1D<PointsPerAxis>>;

} // namespace litepgo::fem
