#pragma once

#include "litepgo/fem/local_kernel.hpp"
#include "litepgo/fem/volume_mapping.hpp"
#include "litepgo/fem/volume_quadrature_cache.hpp"

namespace litepgo {

template <Basis3D Basis, Quadrature3D Quadrature, int NumParameters, PointValue E>
using VolumeFKernel = LocalKernel<VolumeFMapping<Basis, NumParameters>, VolumeQuadratureCache<Basis, Quadrature>, E>;

template <Basis3D Basis, Quadrature3D Quadrature, int NumParameters, PointValue E>
using VolumeFpKernel = LocalKernel<VolumeFpMapping<Basis, NumParameters>, VolumeQuadratureCache<Basis, Quadrature>, E>;

template <Basis3D Basis, Quadrature3D Quadrature, int NumParameters, PointValue E>
using VolumeThetaKernel =
    LocalKernel<VolumeThetaMapping<Basis, NumParameters>, VolumeQuadratureCache<Basis, Quadrature>, E>;

} // namespace litepgo
