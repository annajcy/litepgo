#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo {

// Two copies of a 1D basis; index i*n+j has the last parameter varying fastest.
template <Basis1D B> class TensorProductBasis2D {
  public:
    static constexpr int n = B::num_basis_functions;
    static constexpr int num_basis_functions = n * n;

    using Sample = BasisSample<num_basis_functions, 2>;

    static Sample evaluate(const Eigen::Vector2d& xi) {
        const auto x = B::evaluate(xi.x());
        const auto y = B::evaluate(xi.y());
        Sample result;
        result.N = Eigen::kroneckerProduct(x.N, y.N);
        result.dN_dxi.col(0) = Eigen::kroneckerProduct(x.dN_dxi, y.N);
        result.dN_dxi.col(1) = Eigen::kroneckerProduct(x.N, y.dN_dxi);
        result.d2N_dxi2.col(0) = Eigen::kroneckerProduct(x.d2N_dxi2, y.N);
        result.d2N_dxi2.col(1) = Eigen::kroneckerProduct(x.dN_dxi, y.dN_dxi);
        result.d2N_dxi2.col(2) = Eigen::kroneckerProduct(x.N, y.d2N_dxi2);
        return result;
    }
};

} // namespace litepgo
