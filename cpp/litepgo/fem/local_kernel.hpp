#pragma once

#include "litepgo/fem/local_energy.hpp"
#include "litepgo/fem/volume_quadrature_cache.hpp"

namespace litepgo {

// Integrates densities and DOF derivatives; owns the geometry cache.
class LocalKernel final {
  public:
    explicit LocalKernel(VolumeQuadratureCache cache);

    double value(const LocalValue& energy) const;
    Eigen::VectorXd gradient(const LocalGradient& energy) const;
    Eigen::MatrixXd hessian(const LocalHessian& energy) const;

  private:
    VolumeQuadratureCache cache_;
};

} // namespace litepgo
