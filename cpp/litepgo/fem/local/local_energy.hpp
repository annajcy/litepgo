#pragma once

#include "litepgo/fem/geometry/volume_point_data.hpp"

namespace litepgo {

class LocalEnergy {
  public:
    virtual ~LocalEnergy() = default;
    virtual int num_dofs() const = 0;
    virtual double value(const VolumePointData& point) const = 0;

    // Each capability is independent; override both its query and derivative.
    virtual bool has_gradient() const noexcept;
    virtual bool has_hessian() const noexcept;
    virtual Eigen::VectorXd gradient(const VolumePointData& point) const;
    virtual Eigen::MatrixXd hessian(const VolumePointData& point) const;
};

} // namespace litepgo
