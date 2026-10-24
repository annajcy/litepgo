#include "litepgo/fem/interpolated_value_field.hpp"

#include <utility>

namespace litepgo {

InterpolatedValueField::InterpolatedValueField(Eigen::MatrixXd coefficients)
    : interpolation_(std::move(coefficients)) {}

int InterpolatedValueField::num_dofs() const {
    return interpolation_.num_dofs();
}

Eigen::VectorXd InterpolatedValueField::value(const VolumePointData& point) const {
    return interpolation_.value(point);
}

Eigen::MatrixXd InterpolatedValueField::jacobian(const VolumePointData& point) const {
    return interpolation_.value_matrix(point);
}

} // namespace litepgo
