#include "litepgo/fem/lagrange_basis.h"
#include "litepgo/fem/tensor_product_basis.h"

#include <gtest/gtest.h>

#include <array>

namespace {

using Basis = litepgo::TensorProductBasis3D<litepgo::LagrangeBasis>;
using Vec3 = Eigen::Vector3d;

static_assert(Basis::num_basis_functions == 8);

// Explicit tensor order, independent of the evaluator's implementation.
Eigen::Matrix<double, 3, 8> corners() {
    Eigen::Matrix<double, 3, 8> points;
    points << 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    return points;
}

const std::array<Vec3, 7> samples = {{Vec3(0, 0, 0), Vec3(1, 1, 1), Vec3(0.5, 0.5, 0.5), Vec3(0, 0.3, 0.7),
                                      Vec3(0.27, 0.43, 0.61), Vec3(0.81, 0.12, 0.35), Vec3(1.2, -0.4, 0.8)}};

double polynomial(const Vec3& p) {
    const auto [x, y, z] = std::array{p.x(), p.y(), p.z()};
    return 2 + 3 * x - 4 * y + 5 * z + 6 * x * y - 7 * x * z + 8 * y * z + 9 * x * y * z;
}

Vec3 polynomial_gradient(const Vec3& p) {
    const auto [x, y, z] = std::array{p.x(), p.y(), p.z()};
    return {3 + 6 * y - 7 * z + 9 * y * z, -4 + 6 * x + 8 * z + 9 * x * z, 5 - 7 * x + 8 * y + 9 * x * y};
}

} // namespace

TEST(TensorProductBasis, InterpolatesCornersInTensorOrder) {
    const auto nodes = corners();
    for (int a = 0; a < Basis::num_basis_functions; ++a) {
        const auto sample = Basis::evaluate(nodes.col(a));
        for (int b = 0; b < Basis::num_basis_functions; ++b) {
            EXPECT_DOUBLE_EQ(sample.values[b], a == b ? 1.0 : 0.0);
        }
    }
}

TEST(TensorProductBasis, MatchesLibpgoReferenceHexConvention) {
    // libpgo referenceHex.h and CubicLinearShapeFunction use [0,1]^3 with
    // ring-ordered corners. Its shape gradients are the transpose of ours.
    constexpr std::array<int, 8> hex_to_tensor{0, 4, 6, 2, 1, 5, 7, 3};
    Eigen::Matrix<double, 3, 8> hex_corners;
    hex_corners << 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1;
    const auto tensor_corners = corners();
    for (int c = 0; c < 8; ++c) {
        EXPECT_EQ(tensor_corners.col(hex_to_tensor[c]), hex_corners.col(c));
    }

    const auto sample = Basis::evaluate(Vec3(0.2, 0.3, 0.4));
    Eigen::Matrix<double, 8, 1> weights;
    weights << 0.336, 0.084, 0.036, 0.144, 0.224, 0.056, 0.024, 0.096;
    Eigen::Matrix<double, 8, 3> gradients;
    gradients << -0.42, -0.48, -0.56, 0.42, -0.12, -0.14, 0.18, 0.12, -0.06, -0.18, 0.48, -0.24, -0.28, -0.32, 0.56,
        0.28, -0.08, 0.14, 0.12, 0.08, 0.06, -0.12, 0.32, 0.24;
    for (int c = 0; c < 8; ++c) {
        EXPECT_NEAR(sample.values[hex_to_tensor[c]], weights[c], 1e-15);
        EXPECT_TRUE((sample.gradients.row(hex_to_tensor[c]) - gradients.row(c)).isZero(1e-15));
    }
}

TEST(TensorProductBasis, PartitionsUnityAndHasZeroGradientSum) {
    for (const auto& p : samples) {
        const auto sample = Basis::evaluate(p);
        EXPECT_NEAR(sample.values.sum(), 1.0, 1e-15);
        EXPECT_TRUE(sample.gradients.colwise().sum().isZero(1e-15));
    }
}

TEST(TensorProductBasis, ReproducesAffineVectorFieldsAndJacobians) {
    Eigen::Matrix3d A;
    A << 2, -1, 3, 4, 2, -3, -2, 5, 1;
    const Vec3 b(0.3, -1.7, 2.1);
    const Eigen::Matrix<double, 3, 8> coefficients = (A * corners()).colwise() + b;
    for (const auto& p : samples) {
        const auto sample = Basis::evaluate(p);
        EXPECT_TRUE((coefficients * sample.values).isApprox(A * p + b, 1e-14));
        EXPECT_TRUE((coefficients * sample.gradients).isApprox(A, 1e-14));
    }
}

TEST(TensorProductBasis, ReproducesTrilinearPolynomialsAndGradients) {
    const auto nodes = corners();
    Eigen::Matrix<double, 8, 1> coefficients;
    for (int a = 0; a < Basis::num_basis_functions; ++a) {
        coefficients[a] = polynomial(nodes.col(a));
    }
    for (const auto& p : samples) {
        const auto sample = Basis::evaluate(p);
        EXPECT_NEAR(coefficients.dot(sample.values), polynomial(p), 2e-14);
        EXPECT_TRUE((sample.gradients.transpose() * coefficients).isApprox(polynomial_gradient(p), 1e-14));
    }
}

TEST(TensorProductBasis, GradientsMatchFiniteDifferencesIncludingCorners) {
    constexpr double h = 1e-6;
    for (const auto& p : samples) {
        const auto sample = Basis::evaluate(p);
        for (int d = 0; d < 3; ++d) {
            Vec3 plus = p;
            Vec3 minus = p;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Matrix<double, 8, 1> derivative =
                (Basis::evaluate(plus).values - Basis::evaluate(minus).values) / (2.0 * h);
            EXPECT_TRUE((derivative - sample.gradients.col(d)).isZero(3e-10));
        }
    }
}
