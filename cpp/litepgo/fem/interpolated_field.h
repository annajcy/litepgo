#pragma once

#include "litepgo/fem/volume_reference_geometry.h"

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
    using Values = typename VolumeReferenceGeometry<BasisType>::Values;
    using Gradients = typename VolumeReferenceGeometry<BasisType>::Gradients;

    struct Sample {
        Value value;
        Gradient gradient; // d(field)/dX in the physical reference configuration
        Values basis_values;
        Gradients basis_gradients; // dN/dX, in the same order as the coefficients

        // value = value_matrix() * flatten_rows(coefficients).
        // Column i * Components + c differentiates coefficient (i, c).
        ValueMatrix value_matrix() const {
            return Eigen::kroneckerProduct(basis_values.transpose(),
                                           Eigen::Matrix<double, Components, Components>::Identity());
        }

        // vec(gradient) = gradient_matrix() * flatten_rows(coefficients).
        // vec(gradient) is column-major; Jacobian columns use i * Components + c.
        GradientMatrix gradient_matrix() const {
            return Eigen::kroneckerProduct(basis_gradients.transpose(),
                                           Eigen::Matrix<double, Components, Components>::Identity());
        }
    };

    explicit InterpolatedField(const LocalCoefficients& coefficients) : coefficients_(coefficients) {
        if (!coefficients_.allFinite()) {
            throw std::invalid_argument("InterpolatedField requires finite coefficients");
        }
    }

    Sample evaluate(const ReferencePoint& point) const {
        Sample result{coefficients_.transpose() * point.basis_values, coefficients_.transpose() * point.basis_gradients,
                      point.basis_values, point.basis_gradients};
        if (!result.value.allFinite() || !result.gradient.allFinite()) {
            throw std::invalid_argument("InterpolatedField::evaluate: non-finite field value or gradient");
        }
        return result;
    }

  private:
    LocalCoefficients coefficients_;
};

} // namespace litepgo
