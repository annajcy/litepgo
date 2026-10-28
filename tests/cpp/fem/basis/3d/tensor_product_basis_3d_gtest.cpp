#include "litepgo/fem/basis/1d/linear_lagrange_basis.hpp"
#include "litepgo/fem/basis/3d/tensor_product_basis_3d.hpp"

#include <gtest/gtest.h>

#include <array>

namespace {

static_assert(litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::num_basis_functions == 8);

// Explicit tensor order, independent of the evaluator's implementation.
Eigen::Matrix<double, 3, 8> corners() {
    Eigen::Matrix<double, 3, 8> points;
    points << 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    return points;
}

const std::array<Eigen::Vector3d, 7> samples = {
    {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(0, 0.3, 0.7),
     Eigen::Vector3d(0.27, 0.43, 0.61), Eigen::Vector3d(0.81, 0.12, 0.35), Eigen::Vector3d(1.2, -0.4, 0.8)}};

double polynomial(const Eigen::Vector3d& p) {
    const auto [x, y, z] = std::array{p.x(), p.y(), p.z()};
    return 2 + 3 * x - 4 * y + 5 * z + 6 * x * y - 7 * x * z + 8 * y * z + 9 * x * y * z;
}

Eigen::Vector3d polynomial_gradient(const Eigen::Vector3d& p) {
    const auto [x, y, z] = std::array{p.x(), p.y(), p.z()};
    return {3 + 6 * y - 7 * z + 9 * y * z, -4 + 6 * x + 8 * z + 9 * x * z, 5 - 7 * x + 8 * y + 9 * x * y};
}

} // namespace

TEST(TensorProductBasis3D, InterpolatesCornersInTensorOrder) {
    const auto nodes = corners();
    for (int a = 0; a < litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::num_basis_functions; ++a) {
        const auto basis_sample = litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(nodes.col(a));
        for (int b = 0; b < litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::num_basis_functions; ++b) {
            EXPECT_DOUBLE_EQ(basis_sample.N[b], a == b ? 1.0 : 0.0);
        }
    }
}

TEST(TensorProductBasis3D, PartitionsUnityAndHasZeroGradientSum) {
    for (const auto& p : samples) {
        const auto basis_sample = litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(p);
        EXPECT_NEAR(basis_sample.N.sum(), 1.0, 1e-15);
        EXPECT_TRUE(basis_sample.dN_dxi.colwise().sum().isZero(1e-15));
        EXPECT_TRUE(basis_sample.d2N_dxi2.colwise().sum().isZero(1e-15));
    }
}

TEST(TensorProductBasis3D, ReproducesAffineVectorFieldsAndJacobians) {
    Eigen::Matrix3d A;
    A << 2, -1, 3, 4, 2, -3, -2, 5, 1;
    const Eigen::Vector3d b(0.3, -1.7, 2.1);
    const Eigen::Matrix<double, 3, 8> coefficients = (A * corners()).colwise() + b;
    for (const auto& p : samples) {
        const auto basis_sample = litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(p);
        EXPECT_TRUE((coefficients * basis_sample.N).isApprox(A * p + b, 1e-14));
        EXPECT_TRUE((coefficients * basis_sample.dN_dxi).isApprox(A, 1e-14));
        EXPECT_TRUE((coefficients * basis_sample.d2N_dxi2).isZero(1e-14));
    }
}

TEST(TensorProductBasis3D, ReproducesTrilinearPolynomialsAndGradients) {
    const auto nodes = corners();
    Eigen::Matrix<double, 8, 1> coefficients;
    for (int a = 0; a < litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::num_basis_functions; ++a) {
        coefficients[a] = polynomial(nodes.col(a));
    }
    for (const auto& p : samples) {
        const auto basis_sample = litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(p);
        EXPECT_NEAR(coefficients.dot(basis_sample.N), polynomial(p), 2e-14);
        EXPECT_TRUE((basis_sample.dN_dxi.transpose() * coefficients).isApprox(polynomial_gradient(p), 1e-14));
        Eigen::Matrix<double, 6, 1> expected;
        expected << 0, 6 + 9 * p.z(), -7 + 9 * p.y(), 0, 8 + 9 * p.x(), 0;
        EXPECT_TRUE((basis_sample.d2N_dxi2.transpose() * coefficients).isApprox(expected, 1e-14));
    }
}

TEST(TensorProductBasis3D, GradientsMatchFiniteDifferencesIncludingCorners) {
    constexpr double h = 1e-6;
    for (const auto& p : samples) {
        const auto basis_sample = litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(p);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = p;
            Eigen::Vector3d minus = p;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Matrix<double, 8, 1> derivative =
                (litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(plus).N -
                 litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(minus).N) /
                (2.0 * h);
            EXPECT_TRUE((derivative - basis_sample.dN_dxi.col(d)).isZero(3e-10));
            const Eigen::Matrix<double, 8, 3> second_fd =
                (litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(plus).dN_dxi -
                 litepgo::TensorProductBasis3D<litepgo::LinearLagrangeBasis>::evaluate(minus).dN_dxi) /
                (2 * h);
            constexpr int columns[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
            for (int e = 0; e < 3; ++e)
                EXPECT_TRUE((second_fd.col(e) - basis_sample.d2N_dxi2.col(columns[d][e])).isZero(3e-10));
        }
    }
}
