#include "litepgo/fem/field_interpolation.hpp"

#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

FieldInterpolation::FieldInterpolation(Eigen::MatrixXd coefficients) : coefficients_(std::move(coefficients)) {
    if (coefficients_.rows() == 0 || coefficients_.cols() == 0 ||
        coefficients_.size() > std::numeric_limits<int>::max() || !coefficients_.allFinite())
        throw std::invalid_argument("FieldInterpolation requires nonempty finite coefficients with int-sized DOFs");
}

int FieldInterpolation::num_dofs() const {
    return static_cast<int>(coefficients_.size());
}

int FieldInterpolation::num_components() const {
    return static_cast<int>(coefficients_.cols());
}

Eigen::MatrixXd FieldInterpolation::value_matrix(const VolumePointData& point) const {
    check_value_point(point);
    const int components = num_components();
    return Eigen::kroneckerProduct(point.basis_values.transpose(), Eigen::MatrixXd::Identity(components, components));
}

Eigen::MatrixXd FieldInterpolation::gradient_matrix(const VolumePointData& point) const {
    check_gradient_point(point);
    const int components = num_components();
    return Eigen::kroneckerProduct(point.basis_gradients.transpose(),
                                   Eigen::MatrixXd::Identity(components, components));
}

Eigen::VectorXd FieldInterpolation::value(const VolumePointData& point) const {
    check_value_point(point);
    Eigen::VectorXd result = coefficients_.transpose() * point.basis_values;
    if (!result.allFinite())
        throw std::invalid_argument("FieldInterpolation::value: non-finite result");
    return result;
}

Eigen::Matrix<double, Eigen::Dynamic, 3> FieldInterpolation::gradient(const VolumePointData& point) const {
    check_gradient_point(point);
    Eigen::Matrix<double, Eigen::Dynamic, 3> result = coefficients_.transpose() * point.basis_gradients;
    if (!result.allFinite())
        throw std::invalid_argument("FieldInterpolation::gradient: non-finite result");
    return result;
}

void FieldInterpolation::check_value_point(const VolumePointData& point) const {
    if (point.basis_values.size() != coefficients_.rows() || !point.basis_values.allFinite())
        throw std::invalid_argument("FieldInterpolation: incompatible or non-finite basis values");
}

void FieldInterpolation::check_gradient_point(const VolumePointData& point) const {
    if (point.basis_gradients.rows() != coefficients_.rows() || !point.basis_gradients.allFinite())
        throw std::invalid_argument("FieldInterpolation: incompatible or non-finite basis gradients");
}

} // namespace litepgo
