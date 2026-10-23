#pragma once

#include "litepgo/fem/basis.h"

#include <Eigen/Core>
#include <Eigen/LU>
#include <unsupported/Eigen/KroneckerProduct>

#include <cmath>
#include <stdexcept>

namespace litepgo {

// One isoparametric volume element, independent of any integration rule.
// Geometry and displacement coefficients follow the same basis order.
template <Basis3D BasisType> class VolumeReferenceGeometry final {
  public:
    static constexpr int num_basis_functions = BasisType::num_basis_functions;
    static constexpr int num_dofs = 3 * num_basis_functions;

    // Row i is the three-dimensional geometry coefficient of basis function i.
    // For nodal Q1 elements these are the undeformed vertex positions.
    using GeometryCoefficients = Eigen::Matrix<double, num_basis_functions, 3>;
    using Values = Eigen::Matrix<double, num_basis_functions, 1>;
    using Gradients = Eigen::Matrix<double, num_basis_functions, 3>;
    using DisplacementMatrix = Eigen::Matrix<double, 3, num_dofs>;
    using DeformationGradientMatrix = Eigen::Matrix<double, 9, num_dofs>;

    struct PointData {
        Values values;
        // Row i is dN_i/dX in physical reference coordinates, not parameters xi.
        Gradients gradients;
        double jacobian_determinant;

        // u = B*a, where a is [u0.x,u0.y,u0.z, ...].
        DisplacementMatrix B() const {
            return Eigen::kroneckerProduct(values.transpose(), Eigen::Matrix3d::Identity());
        }

        // vec(F) = vec(I) + BF*a. Matrices are vectorized column by column.
        DeformationGradientMatrix BF() const {
            return Eigen::kroneckerProduct(gradients.transpose(), Eigen::Matrix3d::Identity());
        }
    };

    // Copies the coefficients. Non-finite inputs throw std::invalid_argument.
    // Jacobian validity is checked when evaluating a point, not at construction.
    explicit VolumeReferenceGeometry(const GeometryCoefficients& reference_geometry)
        : reference_geometry_(reference_geometry) {
        if (!reference_geometry_.allFinite()) {
            throw std::invalid_argument("VolumeReferenceGeometry requires finite reference geometry");
        }
    }

    // Does not clamp parameters or cache results. Throws std::invalid_argument
    // for non-finite parameters/data or det(J0) <= 0 at the requested point.
    PointData evaluate(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite()) {
            throw std::invalid_argument("VolumeReferenceGeometry::evaluate requires finite parameters");
        }
        const auto basis = BasisType::evaluate(xi);
        const Eigen::Matrix3d J0 = reference_geometry_.transpose() * basis.gradients;
        const double determinant = J0.determinant();
        if (!J0.allFinite() || !std::isfinite(determinant) || determinant <= 0.0) {
            throw std::invalid_argument("VolumeReferenceGeometry::evaluate: invalid reference Jacobian");
        }

        PointData result{basis.values, basis.gradients * J0.inverse(), determinant};
        if (!result.values.allFinite() || !result.gradients.allFinite()) {
            throw std::invalid_argument("VolumeReferenceGeometry::evaluate: non-finite reference data");
        }
        return result;
    }

  private:
    GeometryCoefficients reference_geometry_;
};

} // namespace litepgo
