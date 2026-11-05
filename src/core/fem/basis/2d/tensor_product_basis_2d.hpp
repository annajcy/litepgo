#pragma once

#include "litepgo/fem/basis/detail/tensor_product_indices.hpp"

namespace litepgo::fem {

// Tensor products in reference-node then derivative-slot order, shared by every operation.
template <Basis1D B> class TensorProductBasis2D {
  public:
    static constexpr int n = B::num_basis_functions;
    static constexpr int num_basis_functions = n * n;

    static CoefficientInfo<2> coefficient_info(int i) {
        if (i < 0 || i >= num_basis_functions)
            throw std::out_of_range("TensorProductBasis2D::coefficient_info: index out of range");
        const auto& index = detail::tensor_product_indices<B, 2>()[static_cast<std::size_t>(i)];
        CoefficientInfo<2> result;
        for (int d = 0; d < 2; ++d) {
            const auto info = B::coefficient_info(index[d]);
            result.point[d] = info.point;
            result.derivative_order[d] = info.derivative_order[0];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector2d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y());
        Eigen::Matrix<double, num_basis_functions, 1> result;
        int row = 0;
        for (const auto& index : detail::tensor_product_indices<B, 2>())
            result[row++] = x[index[0]] * y[index[1]];
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 2> eval_dN_dxi(const Eigen::Vector2d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y());
        Eigen::Matrix<double, num_basis_functions, 2> result;
        int row = 0;
        for (const auto& index : detail::tensor_product_indices<B, 2>()) {
            const auto [i, j] = index;
            result.row(row++) << dx[i] * y[j], x[i] * dy[j];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 3> eval_d2N_dxi2(const Eigen::Vector2d& xi) {
        const auto x = B::eval_N(xi.x()), y = B::eval_N(xi.y());
        const auto dx = B::eval_dN_dxi(xi.x()), dy = B::eval_dN_dxi(xi.y());
        const auto d2x = B::eval_d2N_dxi2(xi.x()), d2y = B::eval_d2N_dxi2(xi.y());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        int row = 0;
        for (const auto& index : detail::tensor_product_indices<B, 2>()) {
            const auto [i, j] = index;
            result.row(row++) << d2x[i] * y[j], dx[i] * dy[j], x[i] * d2y[j];
        }
        return result;
    }
};

} // namespace litepgo::fem
