#pragma once

#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/geometry/volume_quadrature_cache.hpp"

namespace litepgo {

// Integrates densities and DOF derivatives; owns the geometry cache.
class LocalKernel final {
  public:
    explicit LocalKernel(VolumeQuadratureCache cache);

    double value(const LocalEnergy& energy) const;
    Eigen::VectorXd gradient(const LocalEnergy& energy) const;
    Eigen::MatrixXd hessian(const LocalEnergy& energy) const;

  private:
    VolumeQuadratureCache cache_;
};

} // namespace litepgo
