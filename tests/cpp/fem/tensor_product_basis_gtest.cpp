#include "litepgo/fem/lagrange_basis.hpp"
#include "litepgo/fem/tensor_product_basis.hpp"

#include <gtest/gtest.h>

#include <array>

namespace {

static_assert(litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::num_basis_functions == 8);

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

TEST(TensorProductBasis, InterpolatesCornersInTensorOrder) {
    const auto nodes = corners();
    for (int a = 0; a < litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::num_basis_functions; ++a) {
        const auto sample = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::evaluate(nodes.col(a));
        for (int b = 0; b < litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::num_basis_functions; ++b) {
            EXPECT_DOUBLE_EQ(sample.values[b], a == b ? 1.0 : 0.0);
        }
    }
}

TEST(TensorProductBasis, PartitionsUnityAndHasZeroGradientSum) {
    for (const auto& p : samples) {
        const auto sample = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::evaluate(p);
        EXPECT_NEAR(sample.values.sum(), 1.0, 1e-15);
        EXPECT_TRUE(sample.gradients.colwise().sum().isZero(1e-15));
    }
}

TEST(TensorProductBasis, ReproducesAffineVectorFieldsAndJacobians) {
    Eigen::Matrix3d A;
    A << 2, -1, 3, 4, 2, -3, -2, 5, 1;
    const Eigen::Vector3d b(0.3, -1.7, 2.1);
    const Eigen::Matrix<double, 3, 8> coefficients = (A * corners()).colwise() + b;
    for (const auto& p : samples) {
        const auto sample = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::evaluate(p);
        EXPECT_TRUE((coefficients * sample.values).isApprox(A * p + b, 1e-14));
        EXPECT_TRUE((coefficients * sample.gradients).isApprox(A, 1e-14));
    }
}

TEST(TensorProductBasis, ReproducesTrilinearPolynomialsAndGradients) {
    const auto nodes = corners();
    Eigen::Matrix<double, 8, 1> coefficients;
    for (int a = 0; a < litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::num_basis_functions; ++a) {
        coefficients[a] = polynomial(nodes.col(a));
    }
    for (const auto& p : samples) {
        const auto sample = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::evaluate(p);
        EXPECT_NEAR(coefficients.dot(sample.values), polynomial(p), 2e-14);
        EXPECT_TRUE((sample.gradients.transpose() * coefficients).isApprox(polynomial_gradient(p), 1e-14));
    }
}

TEST(TensorProductBasis, GradientsMatchFiniteDifferencesIncludingCorners) {
    constexpr double h = 1e-6;
    for (const auto& p : samples) {
        const auto sample = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::evaluate(p);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = p;
            Eigen::Vector3d minus = p;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Matrix<double, 8, 1> derivative =
                (litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::evaluate(plus).values -
                 litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>::evaluate(minus).values) /
                (2.0 * h);
            EXPECT_TRUE((derivative - sample.gradients.col(d)).isZero(3e-10));
        }
    }
}
