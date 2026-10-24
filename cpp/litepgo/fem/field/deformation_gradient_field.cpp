#include "litepgo/fem/field/deformation_gradient_field.hpp"

#include <stdexcept>
#include <utility>

namespace litepgo {

DeformationGradientField::DeformationGradientField(Eigen::MatrixXd displacement)
    : interpolation_(std::move(displacement)) {
    if (interpolation_.num_components() != 3)
        throw std::invalid_argument("DeformationGradientField requires three displacement components");
}

int DeformationGradientField::num_dofs() const {
    return interpolation_.num_dofs();
}

Eigen::Matrix3d DeformationGradientField::value(const VolumePointData& point) const {
    return Eigen::Matrix3d::Identity() + interpolation_.gradient(point);
}

Eigen::MatrixXd DeformationGradientField::jacobian(const VolumePointData& point) const {
    return interpolation_.gradient_matrix(point);
}

} // namespace litepgo
