#pragma once

#include "litepgo/fem/basis/1d/cubic_hermite_basis.hpp"

namespace litepgo::fem {

// Tricubic Hermite in Hex corner order. Per corner: u, u_xi, u_eta, u_zeta,
// u_xi_eta, u_xi_zeta, u_eta_zeta, u_xi_eta_zeta; no physical length scaling.
class HexHermiteBasis {
  public:
    static constexpr int num_basis_functions = 64;

    // Coefficient index = 8 * corner + derivative slot.
    static CoefficientInfo<3> coefficient_info(int i) {
        if (i < 0 || i >= num_basis_functions)
            throw std::out_of_range("HexHermiteBasis::coefficient_info: index out of range");
        const auto& corner = corners_[i / 8];
        return {Eigen::Vector3d(corner[0], corner[1], corner[2]), derivative_orders_[i % 8]};
    }

    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector3d& xi) {
        const auto x = CubicHermiteBasis::eval_N(xi.x()), y = CubicHermiteBasis::eval_N(xi.y()),
                   z = CubicHermiteBasis::eval_N(xi.z());
        Eigen::Matrix<double, num_basis_functions, 1> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto& corner = corners_[row / 8];
            const auto& order = derivative_orders_[row % 8];
            // CubicHermiteBasis uses index 2 * endpoint + derivative order on each axis.
            const int i = 2 * corner[0] + order[0], j = 2 * corner[1] + order[1], k = 2 * corner[2] + order[2];
            result[row] = x[i] * y[j] * z[k];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 3> eval_dN_dxi(const Eigen::Vector3d& xi) {
        const auto x = CubicHermiteBasis::eval_N(xi.x()), y = CubicHermiteBasis::eval_N(xi.y()),
                   z = CubicHermiteBasis::eval_N(xi.z());
        const auto dx = CubicHermiteBasis::eval_dN_dxi(xi.x()), dy = CubicHermiteBasis::eval_dN_dxi(xi.y()),
                   dz = CubicHermiteBasis::eval_dN_dxi(xi.z());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto& corner = corners_[row / 8];
            const auto& order = derivative_orders_[row % 8];
            const int i = 2 * corner[0] + order[0], j = 2 * corner[1] + order[1], k = 2 * corner[2] + order[2];
            result.row(row) << dx[i] * y[j] * z[k], x[i] * dy[j] * z[k], x[i] * y[j] * dz[k];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 6> eval_d2N_dxi2(const Eigen::Vector3d& xi) {
        const auto x = CubicHermiteBasis::eval_N(xi.x()), y = CubicHermiteBasis::eval_N(xi.y()),
                   z = CubicHermiteBasis::eval_N(xi.z());
        const auto dx = CubicHermiteBasis::eval_dN_dxi(xi.x()), dy = CubicHermiteBasis::eval_dN_dxi(xi.y()),
                   dz = CubicHermiteBasis::eval_dN_dxi(xi.z());
        const auto d2x = CubicHermiteBasis::eval_d2N_dxi2(xi.x()), d2y = CubicHermiteBasis::eval_d2N_dxi2(xi.y()),
                   d2z = CubicHermiteBasis::eval_d2N_dxi2(xi.z());
        Eigen::Matrix<double, num_basis_functions, 6> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto& corner = corners_[row / 8];
            const auto& order = derivative_orders_[row % 8];
            const int i = 2 * corner[0] + order[0], j = 2 * corner[1] + order[1], k = 2 * corner[2] + order[2];
            result(row, 0) = d2x[i] * y[j] * z[k]; // xx
            result(row, 1) = dx[i] * dy[j] * z[k]; // xy
            result(row, 2) = dx[i] * y[j] * dz[k]; // xz
            result(row, 3) = x[i] * d2y[j] * z[k]; // yy
            result(row, 4) = x[i] * dy[j] * dz[k]; // yz
            result(row, 5) = x[i] * y[j] * d2z[k]; // zz
        }
        return result;
    }

  private:
    static constexpr std::array<int, 3> corners_[] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                                      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    static constexpr std::array<int, 3> derivative_orders_[] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                                                                {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
};

} // namespace litepgo::fem
