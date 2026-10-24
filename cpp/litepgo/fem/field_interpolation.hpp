#pragma once

#include "litepgo/fem/volume_reference_geometry.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

#include <stdexcept>

namespace litepgo {

// Coefficient rows follow basis order; columns are field components.
// PointData must be evaluated from the reference geometry of these coefficients.
template <Basis3D Basis, int Components>
    requires(Components > 0)
class FieldInterpolation final {
  public:
    static constexpr int num_basis_functions = Basis::num_basis_functions;
    static constexpr int num_dofs = Components * num_basis_functions;

    // value = value_matrix(point) * flatten_rows(coefficients).
    // Column i * Components + c differentiates coefficient (i, c).
    static Eigen::Matrix<double, Components, num_dofs>
    value_matrix(const typename VolumeReferenceGeometry<Basis>::PointData& point) {
        return Eigen::kroneckerProduct(point.basis_values.transpose(),
                                       Eigen::Matrix<double, Components, Components>::Identity());
    }

    // vec(gradient) is column-major; Jacobian columns use i * Components + c.
    static Eigen::Matrix<double, 3 * Components, num_dofs>
    gradient_matrix(const typename VolumeReferenceGeometry<Basis>::PointData& point) {
        return Eigen::kroneckerProduct(point.basis_gradients.transpose(),
                                       Eigen::Matrix<double, Components, Components>::Identity());
    }

    explicit FieldInterpolation(const Eigen::Matrix<double, num_basis_functions, Components>& coefficients)
        : coefficients_(coefficients) {
        if (!coefficients_.allFinite()) {
            throw std::invalid_argument("FieldInterpolation requires finite coefficients");
        }
    }

    Eigen::Matrix<double, Components, 1> value(const typename VolumeReferenceGeometry<Basis>::PointData& point) const {
        const Eigen::Matrix<double, Components, 1> result = coefficients_.transpose() * point.basis_values;
        if (!result.allFinite()) {
            throw std::invalid_argument("FieldInterpolation::value: non-finite field value");
        }
        return result;
    }

    // Spatial derivative with respect to physical reference coordinates X.
    Eigen::Matrix<double, Components, 3>
    gradient(const typename VolumeReferenceGeometry<Basis>::PointData& point) const {
        const Eigen::Matrix<double, Components, 3> result = coefficients_.transpose() * point.basis_gradients;
        if (!result.allFinite()) {
            throw std::invalid_argument("FieldInterpolation::gradient: non-finite field gradient");
        }
        return result;
    }

  private:
    Eigen::Matrix<double, num_basis_functions, Components> coefficients_;
};

} // namespace litepgo
