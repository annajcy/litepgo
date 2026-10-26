#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo {

// Three copies of a 1D basis. Tensor indices are
// flattened as (i*n + j)*n + k (the last parameter varies fastest, as in libpgo).
template <Basis1D B> class TensorProductBasis3D {
  public:
    static constexpr int n = B::num_basis_functions;
    static constexpr int num_basis_functions = n * n * n;

    struct Sample {
        Eigen::Matrix<double, num_basis_functions, 1> N;
        // Row a contains dN_a/dxi, dN_a/deta, dN_a/dzeta.
        Eigen::Matrix<double, num_basis_functions, 3> dN_dxi;
    };

    static Sample evaluate(const Eigen::Vector3d& xi) {
        const auto x = B::evaluate(xi.x());
        const auto y = B::evaluate(xi.y());
        const auto z = B::evaluate(xi.z());

        Sample result;
        result.N = tensor_product(x.N, y.N, z.N);
        result.dN_dxi.col(0) = tensor_product(x.dN_dxi, y.N, z.N);
        result.dN_dxi.col(1) = tensor_product(x.N, y.dN_dxi, z.N);
        result.dN_dxi.col(2) = tensor_product(x.N, y.N, z.dN_dxi);
        return result;
    }

  private:
    static Eigen::Matrix<double, num_basis_functions, 1> tensor_product(const Eigen::Matrix<double, n, 1>& x,
                                                                        const Eigen::Matrix<double, n, 1>& y,
                                                                        const Eigen::Matrix<double, n, 1>& z) {
        // x (outermost), y, z (innermost) gives the z-fastest tensor ordering.
        return Eigen::kroneckerProduct(x, Eigen::kroneckerProduct(y, z));
    }
};

} // namespace litepgo
