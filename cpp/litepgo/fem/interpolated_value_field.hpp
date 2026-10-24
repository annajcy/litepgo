#pragma once

#include "litepgo/fem/field.hpp"
#include "litepgo/fem/field_interpolation.hpp"

namespace litepgo {

// Owns coefficients; returns the interpolated vector value.
class InterpolatedValueField final : public AffineField<Eigen::VectorXd> {
  public:
    explicit InterpolatedValueField(Eigen::MatrixXd coefficients);
    int num_dofs() const override;
    Eigen::VectorXd value(const VolumePointData& point) const override;
    Eigen::MatrixXd jacobian(const VolumePointData& point) const override;

  private:
    FieldInterpolation interpolation_;
};

} // namespace litepgo
