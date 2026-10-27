#pragma once

#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/field/field.hpp"

#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

// Scalar coefficients flatten by row: m*a+c. Reference derivatives have m rows and three columns.
template <Basis3D B> class InterpolatedField final : public Field {
  public:
    static constexpr int num_basis_functions = B::num_basis_functions;
    explicit InterpolatedField(Eigen::MatrixXd coeffs) : coeffs_(std::move(coeffs)) {
        if (coeffs_.rows() != num_basis_functions || coeffs_.cols() == 0 ||
            coeffs_.size() > std::numeric_limits<int>::max() || !coeffs_.allFinite())
            throw std::invalid_argument(
                "InterpolatedField requires compatible finite coefficients with int-sized DOFs");
    }
    int num_dofs() const override {
        return static_cast<int>(coeffs_.size());
    }
    int num_components() const override {
        return static_cast<int>(coeffs_.cols());
    }

    Eigen::VectorXd value(const Eigen::Vector3d& xi) const override {
        const auto N = checked_basis_sample(xi).N;
        Eigen::VectorXd result = coeffs_.transpose() * N;
        if (!N.allFinite() || !result.allFinite())
            throw std::invalid_argument("InterpolatedField::value: non-finite data");
        return result;
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d& xi) const override {
        const auto dN_dxi = checked_basis_sample(xi).dN_dxi;
        Eigen::Matrix<double, Eigen::Dynamic, 3> result = coeffs_.transpose() * dN_dxi;
        if (!dN_dxi.allFinite() || !result.allFinite())
            throw std::invalid_argument("InterpolatedField::dvalue_dxi: non-finite data");
        return result;
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d& xi) const override {
        const auto N = checked_basis_sample(xi).N;
        if (!N.allFinite())
            throw std::invalid_argument("InterpolatedField::dvalue_dcoeffs: non-finite basis values");
        return Eigen::kroneckerProduct(N.transpose(), Eigen::MatrixXd::Identity(num_components(), num_components()));
    }
    Eigen::MatrixXd dgrad_xi_dcoeffs(const Eigen::Vector3d& xi) const override {
        const auto dN_dxi = checked_basis_sample(xi).dN_dxi;
        if (!dN_dxi.allFinite())
            throw std::invalid_argument("InterpolatedField::dgrad_xi_dcoeffs: non-finite basis gradients");
        return Eigen::kroneckerProduct(dN_dxi.transpose(),
                                       Eigen::MatrixXd::Identity(num_components(), num_components()));
    }

  private:
    static typename B::Sample checked_basis_sample(const Eigen::Vector3d& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("InterpolatedField requires finite parameters");
        return B::evaluate(xi);
    }
    Eigen::MatrixXd coeffs_;
};

} // namespace litepgo
