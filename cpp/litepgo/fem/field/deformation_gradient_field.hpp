#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/field/field_interpolation.hpp"

namespace litepgo {

// Owns displacement coefficients; returns I + grad_X u (F or Fp).
class DeformationGradientField final : public AffineField<Eigen::Matrix3d> {
  public:
    explicit DeformationGradientField(Eigen::MatrixXd displacement);
    int num_dofs() const override;
    Eigen::Matrix3d value(const VolumePointData& point) const override;
    Eigen::MatrixXd jacobian(const VolumePointData& point) const override;

  private:
    FieldInterpolation interpolation_;
};

} // namespace litepgo
