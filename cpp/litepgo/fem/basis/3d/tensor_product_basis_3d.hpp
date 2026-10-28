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

    static Eigen::Matrix<double, num_basis_functions, 1> evaluate_N(const Eigen::Vector3d& xi) {
        return tensor_product(B::evaluate_N(xi.x()), B::evaluate_N(xi.y()), B::evaluate_N(xi.z()));
    }

    static Eigen::Matrix<double, num_basis_functions, 3> evaluate_dN_dxi(const Eigen::Vector3d& xi) {
        const auto x = B::evaluate_N(xi.x()), y = B::evaluate_N(xi.y()), z = B::evaluate_N(xi.z());
        const auto dx = B::evaluate_dN_dxi(xi.x()), dy = B::evaluate_dN_dxi(xi.y()), dz = B::evaluate_dN_dxi(xi.z());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        result.col(0) = tensor_product(dx, y, z);
        result.col(1) = tensor_product(x, dy, z);
        result.col(2) = tensor_product(x, y, dz);
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 6> evaluate_d2N_dxi2(const Eigen::Vector3d& xi) {
        const auto x = B::evaluate_N(xi.x()), y = B::evaluate_N(xi.y()), z = B::evaluate_N(xi.z());
        const auto dx = B::evaluate_dN_dxi(xi.x()), dy = B::evaluate_dN_dxi(xi.y()), dz = B::evaluate_dN_dxi(xi.z());
        const auto d2x = B::evaluate_d2N_dxi2(xi.x()), d2y = B::evaluate_d2N_dxi2(xi.y()),
                   d2z = B::evaluate_d2N_dxi2(xi.z());
        Eigen::Matrix<double, num_basis_functions, 6> result;
        result.col(0) = tensor_product(d2x, y, z);
        result.col(1) = tensor_product(dx, dy, z);
        result.col(2) = tensor_product(dx, y, dz);
        result.col(3) = tensor_product(x, d2y, z);
        result.col(4) = tensor_product(x, dy, dz);
        result.col(5) = tensor_product(x, y, d2z);
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
