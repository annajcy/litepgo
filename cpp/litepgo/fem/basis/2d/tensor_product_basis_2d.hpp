#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo {

// Two copies of a 1D basis; index i*n+j has the last parameter varying fastest.
template <Basis1D B> class TensorProductBasis2D {
  public:
    static constexpr int n = B::num_basis_functions;
    static constexpr int num_basis_functions = n * n;

    struct FirstOrderSample {
        Eigen::Matrix<double, num_basis_functions, 1> N;
        Eigen::Matrix<double, num_basis_functions, 2> dN_dxi;
    };
    struct SecondOrderSample : FirstOrderSample {
        // Columns are (11,12,22), with the mixed derivative stored once, without scaling.
        Eigen::Matrix<double, num_basis_functions, 3> d2N_dxi2;
    };
    using Sample = std::conditional_t<SecondOrderBasis<B, 1>, SecondOrderSample, FirstOrderSample>;

    static Sample evaluate(const Eigen::Vector2d& xi) {
        const auto x = B::evaluate(xi.x());
        const auto y = B::evaluate(xi.y());
        Sample result;
        result.N = Eigen::kroneckerProduct(x.N, y.N);
        result.dN_dxi.col(0) = Eigen::kroneckerProduct(x.dN_dxi, y.N);
        result.dN_dxi.col(1) = Eigen::kroneckerProduct(x.N, y.dN_dxi);
        if constexpr (SecondOrderBasis<B, 1>) {
            result.d2N_dxi2.col(0) = Eigen::kroneckerProduct(x.d2N_dxi2, y.N);
            result.d2N_dxi2.col(1) = Eigen::kroneckerProduct(x.dN_dxi, y.dN_dxi);
            result.d2N_dxi2.col(2) = Eigen::kroneckerProduct(x.N, y.d2N_dxi2);
        }
        return result;
    }
};

} // namespace litepgo
