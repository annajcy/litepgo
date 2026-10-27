#pragma once

#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/3d/tensor_product_quadrature_3d.hpp"

namespace litepgo {

template <int PointsPerAxis> using GaussLegendre3D = TensorProductQuadrature3D<GaussLegendre1D<PointsPerAxis>>;

} // namespace litepgo
