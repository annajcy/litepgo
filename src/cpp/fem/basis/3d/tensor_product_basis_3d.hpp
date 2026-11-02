#pragma once

#include "litepgo/fem/basis/basis.hpp"

#include <Eigen/Core>
#include <unsupported/Eigen/KroneckerProduct>

namespace litepgo::fem {

// Three copies of a 1D basis. Tensor indices are
// flattened as (i*n + j)*n + k (the last parameter varies fastest, as in libpgo).
template <Basis1D B> class TensorProductBasis3D {
  public:
    static constexpr int n = B::num_basis_functions;
    static constexpr int num_basis_functions = n * n * n;

    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector3d& xi) {
        return eval_tensor_product(B::eval_N(xi.x()), B::eval_N(xi.y()), B::eval_N(xi.z()));
    }

    static Eigen::Matrix<double, num_basis_functions, 3> eval_dN_dxi(const Eigen::Vector3d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y()), z = B::eval_N(xi.z());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y()), dz = B::eval_dN_dxi(xi.z());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        result.col(0) = eval_tensor_product(dx, y, z);
        result.col(1) = eval_tensor_product(x, dy, z);
        result.col(2) = eval_tensor_product(x, y, dz);
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 6> eval_d2N_dxi2(const Eigen::Vector3d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y()), z = B::eval_N(xi.z());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y()), dz = B::eval_dN_dxi(xi.z());
        const auto d2x = B::eval_d2N_dxi2(xi.x()), d2y = B::eval_d2N_dxi2(xi.y()), d2z = B::eval_d2N_dxi2(xi.z());
        Eigen::Matrix<double, num_basis_functions, 6> result;
        result.col(0) = eval_tensor_product(d2x, y, z);
        result.col(1) = eval_tensor_product(dx, dy, z);
        result.col(2) = eval_tensor_product(dx, y, dz);
        result.col(3) = eval_tensor_product(x, d2y, z);
        result.col(4) = eval_tensor_product(x, dy, dz);
        result.col(5) = eval_tensor_product(x, y, d2z);
        return result;
    }

  private:
    static Eigen::Matrix<double, num_basis_functions, 1> eval_tensor_product(const Eigen::Matrix<double, n, 1>& x,
                                                                             const Eigen::Matrix<double, n, 1>& y,
                                                                             const Eigen::Matrix<double, n, 1>& z) {
        // x (outermost), y, z (innermost) gives the z-fastest tensor ordering.
        return Eigen::kroneckerProduct(x, Eigen::kroneckerProduct(y, z));
    }
};

} // namespace litepgo::fem
