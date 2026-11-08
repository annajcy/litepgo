#pragma once

#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"

namespace litepgo::fem {

// Trilinear basis in Hex corner order: (0,0,0), (1,0,0), (1,1,0), (0,1,0),
// (0,0,1), (1,0,1), (1,1,1), (0,1,1).
class HexQ1Basis {
  public:
    static constexpr int num_basis_functions = 8;

    static CoefficientInfo<3> coefficient_info(int i) {
        if (i < 0 || i >= num_basis_functions)
            throw std::out_of_range("HexQ1Basis::coefficient_info: index out of range");
        const auto& corner = corners_[i];
        return {Eigen::Vector3d(corner[0], corner[1], corner[2]), {}};
    }

    static Eigen::Matrix<double, num_basis_functions, 1> eval_N(const Eigen::Vector3d& xi) {
        const auto x = LinearLagrangeBasis::eval_N(xi.x()), y = LinearLagrangeBasis::eval_N(xi.y()),
                   z = LinearLagrangeBasis::eval_N(xi.z());
        Eigen::Matrix<double, num_basis_functions, 1> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto [i, j, k] = corners_[row];
            result[row] = x[i] * y[j] * z[k];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 3> eval_dN_dxi(const Eigen::Vector3d& xi) {
        const auto x = LinearLagrangeBasis::eval_N(xi.x()), y = LinearLagrangeBasis::eval_N(xi.y()),
                   z = LinearLagrangeBasis::eval_N(xi.z());
        const auto dx = LinearLagrangeBasis::eval_dN_dxi(xi.x()), dy = LinearLagrangeBasis::eval_dN_dxi(xi.y()),
                   dz = LinearLagrangeBasis::eval_dN_dxi(xi.z());
        Eigen::Matrix<double, num_basis_functions, 3> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto [i, j, k] = corners_[row];
            result.row(row) << dx[i] * y[j] * z[k], x[i] * dy[j] * z[k], x[i] * y[j] * dz[k];
        }
        return result;
    }

    static Eigen::Matrix<double, num_basis_functions, 6> eval_d2N_dxi2(const Eigen::Vector3d& xi) {
        const auto x = LinearLagrangeBasis::eval_N(xi.x()), y = LinearLagrangeBasis::eval_N(xi.y()),
                   z = LinearLagrangeBasis::eval_N(xi.z());
        const auto dx = LinearLagrangeBasis::eval_dN_dxi(xi.x()), dy = LinearLagrangeBasis::eval_dN_dxi(xi.y()),
                   dz = LinearLagrangeBasis::eval_dN_dxi(xi.z());
        Eigen::Matrix<double, num_basis_functions, 6> result;
        for (int row = 0; row < num_basis_functions; ++row) {
            const auto [i, j, k] = corners_[row];
            result(row, 0) = 0;                    // xx
            result(row, 1) = dx[i] * dy[j] * z[k]; // xy
            result(row, 2) = dx[i] * y[j] * dz[k]; // xz
            result(row, 3) = 0;                    // yy
            result(row, 4) = x[i] * dy[j] * dz[k]; // yz
            result(row, 5) = 0;                    // zz
        }
        return result;
    }

  private:
    static constexpr std::array<int, 3> corners_[] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                                      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
};

} // namespace litepgo::fem
