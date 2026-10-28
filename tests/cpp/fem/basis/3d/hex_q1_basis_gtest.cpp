#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

namespace {

using litepgo::HexQ1Basis;

Eigen::Matrix<double, 3, 8> reference_corners() {
    Eigen::Matrix<double, 3, 8> points;
    points << 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1;
    return points;
}

const std::array<Eigen::Vector3d, 6> samples{{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1),
                                              Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(0, 0.3, 0.7),
                                              Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(1.2, -0.1, 0.8)}};

} // namespace

TEST(HexQ1Basis, InterpolatesReferenceNodesInLibpgoOrder) {
    const auto corners = reference_corners();
    for (int a = 0; a < 8; ++a) {
        EXPECT_EQ(HexQ1Basis::reference_node(a), corners.col(a));
        const auto basis_sample = HexQ1Basis::evaluate(corners.col(a));
        for (int b = 0; b < 8; ++b) {
            EXPECT_DOUBLE_EQ(basis_sample.N[b], a == b ? 1.0 : 0.0);
        }
    }
    EXPECT_THROW(HexQ1Basis::reference_node(-1), std::out_of_range);
    EXPECT_THROW(HexQ1Basis::reference_node(8), std::out_of_range);
}

TEST(HexQ1Basis, MatchesVerifiedLibpgoWeightsAndGradients) {
    // Verified against fit-micro's CubicLinearShapeFunction at (0.2,0.3,0.4).
    const auto basis_sample = HexQ1Basis::evaluate(Eigen::Vector3d(0.2, 0.3, 0.4));
    Eigen::Matrix<double, 8, 1> weights;
    weights << 0.336, 0.084, 0.036, 0.144, 0.224, 0.056, 0.024, 0.096;
    Eigen::Matrix<double, 8, 3> gradients;
    gradients << -0.42, -0.48, -0.56, 0.42, -0.12, -0.14, 0.18, 0.12, -0.06, -0.18, 0.48, -0.24, -0.28, -0.32, 0.56,
        0.28, -0.08, 0.14, 0.12, 0.08, 0.06, -0.12, 0.32, 0.24;
    EXPECT_TRUE((basis_sample.N - weights).isZero(1e-15));
    EXPECT_TRUE((basis_sample.dN_dxi - gradients).isZero(1e-15));
}

TEST(HexQ1Basis, PartitionsUnityAndHasZeroGradientSum) {
    for (const auto& p : samples) {
        const auto basis_sample = HexQ1Basis::evaluate(p);
        EXPECT_NEAR(basis_sample.N.sum(), 1.0, 1e-15);
        EXPECT_TRUE(basis_sample.dN_dxi.colwise().sum().isZero(1e-15));
        EXPECT_TRUE(basis_sample.d2N_dxi2.colwise().sum().isZero(1e-15));
    }
}

TEST(HexQ1Basis, ReproducesAffineGeometryAndVectorFields) {
    const auto corners = reference_corners();
    Eigen::Matrix3d dX_dxi;
    dX_dxi << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1);
    Eigen::Matrix3d A;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    const Eigen::Vector3d b(-0.3, 0.4, 0.7);
    const Eigen::Matrix<double, 3, 8> Xe = (dX_dxi * corners).colwise() + origin;
    const Eigen::Matrix<double, 3, 8> Ue = (A * Xe).colwise() + b;

    for (const auto& p : samples) {
        const auto basis_sample = HexQ1Basis::evaluate(p);
        const Eigen::Vector3d X = origin + dX_dxi * p;
        EXPECT_TRUE((Xe * basis_sample.N).isApprox(X, 1e-14));
        EXPECT_TRUE((Xe * basis_sample.dN_dxi).isApprox(dX_dxi, 1e-14));
        EXPECT_TRUE((Ue * basis_sample.N).isApprox(A * X + b, 1e-14));
        EXPECT_TRUE((Ue * basis_sample.dN_dxi).isApprox(A * dX_dxi, 1e-14));
        EXPECT_TRUE((Xe * basis_sample.d2N_dxi2).isZero(1e-14));
        EXPECT_TRUE((Ue * basis_sample.d2N_dxi2).isZero(1e-14));
    }
}

TEST(HexQ1Basis, GradientsMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (const auto& p : samples) {
        const auto basis_sample = HexQ1Basis::evaluate(p);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = p;
            Eigen::Vector3d minus = p;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Matrix<double, 8, 1> derivative =
                (HexQ1Basis::evaluate(plus).N - HexQ1Basis::evaluate(minus).N) / (2.0 * h);
            EXPECT_TRUE((derivative - basis_sample.dN_dxi.col(d)).isZero(3e-10));
            const Eigen::Matrix<double, 8, 3> second_fd =
                (HexQ1Basis::evaluate(plus).dN_dxi - HexQ1Basis::evaluate(minus).dN_dxi) / (2 * h);
            constexpr int columns[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
            for (int e = 0; e < 3; ++e)
                EXPECT_TRUE((second_fd.col(e) - basis_sample.d2N_dxi2.col(columns[d][e])).isZero(3e-10));
        }
    }
}
