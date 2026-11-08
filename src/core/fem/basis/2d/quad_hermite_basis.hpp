#pragma once

#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"

namespace litepgo::fem {

// Bicubic Hermite; corner order (0,0), (1,0), (1,1), (0,1).
// Per corner: u, u_xi, u_eta, u_xi_eta, with reference-coordinate derivatives.
class QuadHermiteBasis {
  public:
    static constexpr int num_basis_functions = 16;

    // Coefficient index = 4 * corner + derivative slot.
    static CoefficientInfo<2> coefficient_info(int i) {
        if (i < 0 || i >= num_basis_functions)
            throw std::out_of_range("QuadHermiteBasis::coefficient_info: index out of range");
        const auto& corner = corners_[i / 4];
        return {Eigen::Vector2d(corner[0], corner[1]), derivative_orders_[i % 4]};
    }

    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector2d& xi) {
        const auto x = CubicHermiteBasis::eval_N(xi.x()), y = CubicHermiteBasis::eval_N(xi.y());
        Eigen::Matrix<double, num_basis_functions, 1> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto& corner = corners_[row / 4];
            const auto& order = derivative_orders_[row % 4];
            // CubicHermiteBasis uses index 2 * endpoint + derivative order on each axis.
            const int i = 2 * corner[0] + order[0], j = 2 * corner[1] + order[1];
            result[row] = x[i] * y[j];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 2> eval_dN_dxi(const Eigen::Vector2d& xi) {
        const auto x = CubicHermiteBasis::eval_N(xi.x()), y = CubicHermiteBasis::eval_N(xi.y());
        const auto dx = CubicHermiteBasis::eval_dN_dxi(xi.x()), dy = CubicHermiteBasis::eval_dN_dxi(xi.y());
        Eigen::Matrix<double, num_basis_functions, 2> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto& corner = corners_[row / 4];
            const auto& order = derivative_orders_[row % 4];
            const int i = 2 * corner[0] + order[0], j = 2 * corner[1] + order[1];
            result.row(row) << dx[i] * y[j], x[i] * dy[j];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 3> eval_d2N_dxi2(const Eigen::Vector2d& xi) {
        const auto x = CubicHermiteBasis::eval_N(xi.x()), y = CubicHermiteBasis::eval_N(xi.y());
        const auto dx = CubicHermiteBasis::eval_dN_dxi(xi.x()), dy = CubicHermiteBasis::eval_dN_dxi(xi.y());
        const auto d2x = CubicHermiteBasis::eval_d2N_dxi2(xi.x()), d2y = CubicHermiteBasis::eval_d2N_dxi2(xi.y());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto& corner = corners_[row / 4];
            const auto& order = derivative_orders_[row % 4];
            const int i = 2 * corner[0] + order[0], j = 2 * corner[1] + order[1];
            result(row, 0) = d2x[i] * y[j]; // xx
            result(row, 1) = dx[i] * dy[j]; // xy
            result(row, 2) = x[i] * d2y[j]; // yy
        }
        return result;
    }

  private:
    static constexpr std::array<int, 2> corners_[] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    static constexpr std::array<int, 2> derivative_orders_[] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
};

} // namespace litepgo::fem
