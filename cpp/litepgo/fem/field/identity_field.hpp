#pragma once

#include "litepgo/fem/field/field.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace litepgo {

// Spatially constant; coefficients and values use the same component vector.
class IdentityField final : public Field {
  public:
    explicit IdentityField(Eigen::VectorXd coeffs) : coeffs_(std::move(coeffs)) {
        if (coeffs_.size() == 0 || coeffs_.size() > std::numeric_limits<int>::max() || !coeffs_.allFinite())
            throw std::invalid_argument("IdentityField requires nonempty finite values with int-sized DOFs");
    }
    int num_dofs() const override {
        return static_cast<int>(coeffs_.size());
    }
    int num_components() const override {
        return static_cast<int>(coeffs_.size());
    }
    Eigen::VectorXd value(const Eigen::Vector3d& xi) const override {
        check_xi(xi);
        return coeffs_;
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d& xi) const override {
        check_xi(xi);
        return Eigen::Matrix<double, Eigen::Dynamic, 3>::Zero(coeffs_.size(), 3);
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d& xi) const override {
        check_xi(xi);
        return Eigen::MatrixXd::Identity(coeffs_.size(), coeffs_.size());
    }
    Eigen::MatrixXd dgrad_xi_dcoeffs(const Eigen::Vector3d& xi) const override {
        check_xi(xi);
        return Eigen::MatrixXd::Zero(3 * coeffs_.size(), coeffs_.size());
    }

  private:
    static void check_xi(const Eigen::Vector3d& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("IdentityField requires finite parameters");
    }
    Eigen::VectorXd coeffs_;
};

} // namespace litepgo
