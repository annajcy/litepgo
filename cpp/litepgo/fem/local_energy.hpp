#pragma once

#include "litepgo/fem/volume_point_data.hpp"

namespace litepgo {

class LocalValue {
  public:
    virtual ~LocalValue() = default;
    virtual double value(const VolumePointData& point) const = 0;
};

// Independent capabilities: computing a Hessian does not require a gradient API.
class LocalGradient {
  public:
    virtual ~LocalGradient() = default;
    virtual int num_dofs() const = 0;
    virtual Eigen::VectorXd gradient(const VolumePointData& point) const = 0;
};

class LocalHessian {
  public:
    virtual ~LocalHessian() = default;
    virtual int num_dofs() const = 0;
    virtual Eigen::MatrixXd hessian(const VolumePointData& point) const = 0;
};

} // namespace litepgo
