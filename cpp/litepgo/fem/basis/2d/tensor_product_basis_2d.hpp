#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo::fem {

// Two copies of a 1D basis; index i*n+j has the last parameter varying fastest.
template <Basis1D B> class TensorProductBasis2D {
  public:
    static constexpr int n = B::num_basis_functions;
    static constexpr int num_basis_functions = n * n;

    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector2d& xi) {
        return Eigen::kroneckerProduct(B::eval_N(xi.x()), B::eval_N(xi.y()));
    }

    static Eigen::Matrix<double, num_basis_functions, 2> eval_dN_dxi(const Eigen::Vector2d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y());
        Eigen::Matrix<double, num_basis_functions, 2> result;
        result.col(0) = Eigen::kroneckerProduct(dx, y);
        result.col(1) = Eigen::kroneckerProduct(x, dy);
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 3> eval_d2N_dxi2(const Eigen::Vector2d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y());
        const auto d2x = B::eval_d2N_dxi2(xi.x()), d2y = B::eval_d2N_dxi2(xi.y());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        result.col(0) = Eigen::kroneckerProduct(d2x, y);
        result.col(1) = Eigen::kroneckerProduct(dx, dy);
        result.col(2) = Eigen::kroneckerProduct(x, d2y);
        return result;
    }
};

} // namespace litepgo::fem
