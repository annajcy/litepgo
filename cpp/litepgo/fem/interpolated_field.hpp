#pragma once

#include "litepgo/fem/volume_reference_geometry.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

#include <stdexcept>

namespace litepgo {

// Coefficient rows follow basis order; columns are field components.
// PointData must be evaluated from the reference geometry of these coefficients.
template <Basis3D BasisType, int Components>
    requires(Components > 0)
class InterpolatedField final {
  public:
    using ReferencePoint = typename VolumeReferenceGeometry<BasisType>::PointData;
    static constexpr int num_basis_functions = BasisType::num_basis_functions;
    static constexpr int num_dofs = Components * num_basis_functions;

    using LocalCoefficients = Eigen::Matrix<double, num_basis_functions, Components>;
    using Value = Eigen::Matrix<double, Components, 1>;
    using Gradient = Eigen::Matrix<double, Components, 3>;
    using ValueMatrix = Eigen::Matrix<double, Components, num_dofs>;
    using GradientMatrix = Eigen::Matrix<double, 3 * Components, num_dofs>;

    // value = value_matrix(point) * flatten_rows(coefficients).
    // Column i * Components + c differentiates coefficient (i, c).
    static ValueMatrix value_matrix(const ReferencePoint& point) {
        return Eigen::kroneckerProduct(point.basis_values.transpose(),
                                       Eigen::Matrix<double, Components, Components>::Identity());
    }

    // vec(gradient) is column-major; Jacobian columns use i * Components + c.
    static GradientMatrix gradient_matrix(const ReferencePoint& point) {
        return Eigen::kroneckerProduct(point.basis_gradients.transpose(),
                                       Eigen::Matrix<double, Components, Components>::Identity());
    }

    explicit InterpolatedField(const LocalCoefficients& coefficients) : coefficients_(coefficients) {
        if (!coefficients_.allFinite()) {
            throw std::invalid_argument("InterpolatedField requires finite coefficients");
        }
    }

    Value value(const ReferencePoint& point) const {
        const Value result = coefficients_.transpose() * point.basis_values;
        if (!result.allFinite()) {
            throw std::invalid_argument("InterpolatedField::value: non-finite field value");
        }
        return result;
    }

    // Spatial derivative with respect to physical reference coordinates X.
    Gradient gradient(const ReferencePoint& point) const {
        const Gradient result = coefficients_.transpose() * point.basis_gradients;
        if (!result.allFinite()) {
            throw std::invalid_argument("InterpolatedField::gradient: non-finite field gradient");
        }
        return result;
    }

  private:
    LocalCoefficients coefficients_;
};

} // namespace litepgo
