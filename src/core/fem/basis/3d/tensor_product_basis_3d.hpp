#pragma once

#include "litepgo/fem/basis/detail/tensor_product_indices.hpp"

namespace litepgo::fem {

// Tensor products in reference-node then derivative-slot order, shared by every operation.
template <Basis1D B> class TensorProductBasis3D {
  public:
    static constexpr int n = B::num_basis_functions;
    static constexpr int num_basis_functions = n * n * n;

    static CoefficientInfo<3> coefficient_info(int i) {
        if (i < 0 || i >= num_basis_functions)
            throw std::out_of_range("TensorProductBasis3D::coefficient_info: index out of range");
        const auto& index = detail::tensor_product_indices<B, 3>()[static_cast<std::size_t>(i)];
        CoefficientInfo<3> result;
        for (int d = 0; d < 3; ++d) {
            const auto info = B::coefficient_info(index[d]);
            result.point[d] = info.point;
            result.derivative_order[d] = info.derivative_order[0];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector3d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y()), z = B::eval_N(xi.z());
        Eigen::Matrix<double, num_basis_functions, 1> result;
        int row = 0;
        for (const auto& index : detail::tensor_product_indices<B, 3>())
            result[row++] = x[index[0]] * y[index[1]] * z[index[2]];
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 3> eval_dN_dxi(const Eigen::Vector3d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y()), z = B::eval_N(xi.z());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y()), dz = B::eval_dN_dxi(xi.z());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        int row = 0;
        for (const auto& index : detail::tensor_product_indices<B, 3>()) {
            const auto [i, j, k] = index;
            result.row(row++) << dx[i] * y[j] * z[k], x[i] * dy[j] * z[k], x[i] * y[j] * dz[k];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 6> eval_d2N_dxi2(const Eigen::Vector3d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y()), z = B::eval_N(xi.z());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y()), dz = B::eval_dN_dxi(xi.z());
        const auto d2x = B::eval_d2N_dxi2(xi.x()), d2y = B::eval_d2N_dxi2(xi.y()), d2z = B::eval_d2N_dxi2(xi.z());
        Eigen::Matrix<double, num_basis_functions, 6> result;
        int row = 0;
        for (const auto& index : detail::tensor_product_indices<B, 3>()) {
            const auto [i, j, k] = index;
            result.row(row++) << d2x[i] * y[j] * z[k], dx[i] * dy[j] * z[k], dx[i] * y[j] * dz[k], x[i] * d2y[j] * z[k],
                x[i] * dy[j] * dz[k], x[i] * y[j] * d2z[k];
        }
        return result;
    }
};

} // namespace litepgo::fem
