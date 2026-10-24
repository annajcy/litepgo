#pragma once

#include "litepgo/fem/basis.hpp"
#include "litepgo/fem/volume_point_data.hpp"

#include <Eigen/Core>
#include <Eigen/LU>

#include <cmath>
#include <stdexcept>

namespace litepgo {

// Maps the parameter domain to one element in the physical reference configuration.
// Independent of the fields interpolated on that element and of any integration rule.
template <Basis3D BasisType> class VolumeReferenceGeometry final {
  public:
    static constexpr int num_basis_functions = BasisType::num_basis_functions;

    // Row i contains the geometry coefficient of basis function i (rest vertex positions for Q1).
    // Copies the coefficients. Non-finite inputs throw std::invalid_argument.
    // Jacobian validity is checked when evaluating a point, not at construction.
    explicit VolumeReferenceGeometry(const Eigen::Matrix<double, num_basis_functions, 3>& reference_geometry)
        : reference_geometry_(reference_geometry) {
        if (!reference_geometry_.allFinite()) {
            throw std::invalid_argument("VolumeReferenceGeometry requires finite reference geometry");
        }
    }

    // Does not clamp parameters or cache results. Throws std::invalid_argument
    // for non-finite parameters/data or det(J0) <= 0 at the requested point.
    VolumePointData evaluate(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite()) {
            throw std::invalid_argument("VolumeReferenceGeometry::evaluate requires finite parameters");
        }
        const auto basis = BasisType::evaluate(xi);
        const Eigen::Vector3d position = reference_geometry_.transpose() * basis.values;
        const Eigen::Matrix3d J0 = reference_geometry_.transpose() * basis.gradients;
        const double determinant = J0.determinant();
        if (!position.allFinite() || !J0.allFinite() || !std::isfinite(determinant) || determinant <= 0.0) {
            throw std::invalid_argument("VolumeReferenceGeometry::evaluate: invalid reference mapping");
        }

        VolumePointData result{position, J0, basis.values, basis.gradients * J0.inverse(), determinant};
        if (!result.basis_values.allFinite() || !result.basis_gradients.allFinite()) {
            throw std::invalid_argument("VolumeReferenceGeometry::evaluate: non-finite reference data");
        }
        return result;
    }

  private:
    Eigen::Matrix<double, num_basis_functions, 3> reference_geometry_;
};

} // namespace litepgo
