#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

// Rows are basis coefficients, columns are components; x_local flattens by row.
// For n basis functions and m components, value has m entries and dvalue_dxi is m-by-3.
template <Basis3D B> class Interpolation final {
  public:
    static constexpr int num_basis_functions = B::num_basis_functions;

    explicit Interpolation(Eigen::MatrixXd coefficients) : coefficients_(std::move(coefficients)) {
        if (coefficients_.rows() != num_basis_functions || coefficients_.cols() == 0 ||
            coefficients_.size() > std::numeric_limits<int>::max() || !coefficients_.allFinite())
            throw std::invalid_argument("Interpolation requires compatible finite coefficients with int-sized DOFs");
    }

    int num_dofs() const {
        return static_cast<int>(coefficients_.size());
    }
    int num_components() const {
        return static_cast<int>(coefficients_.cols());
    }

    Eigen::VectorXd value(const Eigen::Vector3d& xi) const {
        const auto N = basis_sample(xi).N;
        Eigen::VectorXd result = coefficients_.transpose() * N;
        if (!N.allFinite() || !result.allFinite())
            throw std::invalid_argument("Interpolation::value: non-finite data");
        return result;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d& xi) const {
        const auto dN_dxi = basis_sample(xi).dN_dxi;
        Eigen::Matrix<double, Eigen::Dynamic, 3> result = coefficients_.transpose() * dN_dxi;
        if (!dN_dxi.allFinite() || !result.allFinite())
            throw std::invalid_argument("Interpolation::dvalue_dxi: non-finite data");
        return result;
    }

    // m-by-(n*m): derivative of value with respect to row-ordered coefficients.
    Eigen::MatrixXd dvalue_dx_local(const Eigen::Vector3d& xi) const {
        const auto N = basis_sample(xi).N;
        if (!N.allFinite())
            throw std::invalid_argument("Interpolation::dvalue_dx_local: non-finite basis values");
        return Eigen::kroneckerProduct(N.transpose(), Eigen::MatrixXd::Identity(num_components(), num_components()));
    }

    // (3*m)-by-(n*m): derivative of column-major vec(dvalue_dxi) with respect to x_local.
    Eigen::MatrixXd dgrad_xi_dx_local(const Eigen::Vector3d& xi) const {
        const auto dN_dxi = basis_sample(xi).dN_dxi;
        if (!dN_dxi.allFinite())
            throw std::invalid_argument("Interpolation::dgrad_xi_dx_local: non-finite basis gradients");
        return Eigen::kroneckerProduct(dN_dxi.transpose(),
                                       Eigen::MatrixXd::Identity(num_components(), num_components()));
    }

  private:
    static typename B::Sample basis_sample(const Eigen::Vector3d& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("Interpolation requires finite parameters");
        return B::evaluate(xi);
    }

    Eigen::MatrixXd coefficients_;
};

} // namespace litepgo
