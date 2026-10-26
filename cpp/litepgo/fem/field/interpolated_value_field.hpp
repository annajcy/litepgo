#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/geometry/interpolation.hpp"

namespace litepgo {

// Owns coefficients; returns the interpolated vector value.
template <Basis3D B> class InterpolatedValueField final : public AffineField<Eigen::VectorXd> {
  public:
    explicit InterpolatedValueField(Eigen::MatrixXd coefficients) : interpolation_(std::move(coefficients)) {}
    int num_dofs() const override {
        return interpolation_.num_dofs();
    }
    Eigen::VectorXd value(const RestPointData& point) const override {
        return interpolation_.value(point.xi);
    }
    Eigen::MatrixXd dvalue_dx_local(const RestPointData& point) const override {
        return interpolation_.dvalue_dx_local(point.xi);
    }

  private:
    Interpolation<B> interpolation_;
};

} // namespace litepgo
