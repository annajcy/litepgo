#pragma once

#include "litepgo/fem/volume_reference_geometry.h"

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

#include <stdexcept>

namespace litepgo {

// Coefficients are ordered [basis0.component0, ..., basis1.component0, ...].
// PointData must be evaluated from the reference geometry of these coefficients.
template <Basis3D BasisType, int Components>
    requires(Components > 0)
class InterpolatedField final {
  public:
    using ReferencePoint = typename VolumeReferenceGeometry<BasisType>::PointData;
    static constexpr int num_basis_functions = BasisType::num_basis_functions;
    static constexpr int num_dofs = Components * num_basis_functions;

    using LocalCoefficients = Eigen::Matrix<double, num_dofs, 1>;
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

        // value = value_matrix() * coefficients.
        ValueMatrix value_matrix() const {
            return Eigen::kroneckerProduct(basis_values.transpose(),
                                           Eigen::Matrix<double, Components, Components>::Identity());
        }

        // vec(gradient) = gradient_matrix() * coefficients; column-major vec.
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
        const Eigen::Map<const Eigen::Matrix<double, Components, num_basis_functions>> coefficients(
            coefficients_.data());
        Sample result{coefficients * point.basis_values, coefficients * point.basis_gradients, point.basis_values,
                      point.basis_gradients};
        if (!result.value.allFinite() || !result.gradient.allFinite()) {
            throw std::invalid_argument("InterpolatedField::evaluate: non-finite field value or gradient");
        }
        return result;
    }

  private:
    LocalCoefficients coefficients_;
};

} // namespace litepgo
