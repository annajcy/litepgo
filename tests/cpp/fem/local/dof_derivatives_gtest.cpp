#include "litepgo/fem/local/dof_derivatives.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace {
TEST(DofDerivatives, AffineChainRuleMatchesFiniteDifferences) {
    Eigen::Matrix<double, 4, 3> dvalue_dx_local;
    dvalue_dx_local << 1, 0.2, -0.3, 0.1, 2, 0.4, -0.2, 0.3, 0.8, 0.7, -0.4, 1.2;
    Eigen::Matrix4d H = Eigen::Vector4d(1, 2, 3, 4).asDiagonal();
    H(0, 3) = H(3, 0) = 0.2;
    const Eigen::Vector3d a(0.2, -0.3, 0.5);
    const Eigen::Vector4d offset(0.3, 0.1, -0.2, 0.4);
    const auto energy = [&](const Eigen::Vector3d& x) {
        const Eigen::Vector4d y = dvalue_dx_local * x + offset;
        return 0.5 * y.dot(H * y);
    };
    const auto gradient = [&](const Eigen::Vector3d& x) {
        const Eigen::Vector4d g = H * (dvalue_dx_local * x + offset);
        const Eigen::Matrix<double, 2, 2, Eigen::RowMajor> matrix = Eigen::Map<const Eigen::Matrix2d>(g.data());
        return litepgo::gradient_wrt_dofs(dvalue_dx_local, matrix);
    };
    const auto g = gradient(a);
    const auto hessian = litepgo::hessian_wrt_dofs(dvalue_dx_local, H);
    for (double h : {1e-3, 1e-4}) {
        for (int j = 0; j < 3; ++j) {
            auto plus = a, minus = a;
            plus[j] += h;
            minus[j] -= h;
            EXPECT_NEAR(g[j], (energy(plus) - energy(minus)) / (2 * h), 1e-10);
            EXPECT_TRUE(hessian.col(j).isApprox((gradient(plus) - gradient(minus)) / (2 * h), 1e-10));
        }
    }
    // Expressions and temporaries must yield an owning return value.
    const auto copied = litepgo::hessian_wrt_dofs(2 * dvalue_dx_local, H + H);
    EXPECT_TRUE(copied.isApprox(8 * hessian, 1e-13));
}
} // namespace

TEST(DofDerivatives, RejectsMismatchedDynamicSizes) {
    Eigen::MatrixXd dvalue_dx_local = Eigen::MatrixXd::Zero(4, 3);
    EXPECT_THROW(litepgo::gradient_wrt_dofs(dvalue_dx_local, Eigen::Vector3d::Zero()), std::invalid_argument);
    EXPECT_THROW(litepgo::hessian_wrt_dofs(dvalue_dx_local, Eigen::Matrix3d::Zero()), std::invalid_argument);
}

TEST(DofDerivatives, WeightedContractionsAddToBuffersAndPackStridedGradientsByColumn) {
    Eigen::Matrix<double, 4, 3> dvalue_dx_local;
    dvalue_dx_local << 1, 0.2, -0.3, 0.1, 2, 0.4, -0.2, 0.3, 0.8, 0.7, -0.4, 1.2;
    Eigen::Matrix4d H = Eigen::Vector4d(1, 2, 3, 4).asDiagonal();
    H(0, 3) = 0.2;
    H(3, 0) = -0.1;
    const Eigen::MatrixXd storage = Eigen::MatrixXd::Random(5, 4);
    const auto gradient = storage.block<2, 2>(1, 1);
    for (double weight : {0.0, 0.3, -0.7}) {
        Eigen::VectorXd g = Eigen::VectorXd::Constant(3, 2);
        Eigen::MatrixXd h = Eigen::MatrixXd::Constant(3, 3, 4);
        litepgo::add_gradient_wrt_dofs(dvalue_dx_local, gradient, weight, g);
        litepgo::add_hessian_wrt_dofs(dvalue_dx_local, H, weight, h);
        EXPECT_TRUE(g.isApprox(
            Eigen::Vector3d::Constant(2) + weight * litepgo::gradient_wrt_dofs(dvalue_dx_local, gradient), 1e-13));
        EXPECT_TRUE(
            h.isApprox(Eigen::Matrix3d::Constant(4) + weight * litepgo::hessian_wrt_dofs(dvalue_dx_local, H), 1e-13));
    }
}

TEST(DofDerivatives, WeightedContractionsRejectInvalidInputsBeforeWriting) {
    const Eigen::MatrixXd dvalue_dx_local = Eigen::MatrixXd::Ones(4, 3);
    Eigen::VectorXd g = Eigen::VectorXd::Constant(3, 7);
    Eigen::MatrixXd h = Eigen::MatrixXd::Constant(3, 3, 9);
    EXPECT_THROW(litepgo::add_gradient_wrt_dofs(dvalue_dx_local, Eigen::Vector3d::Ones(), 1, g), std::invalid_argument);
    EXPECT_THROW(litepgo::add_hessian_wrt_dofs(dvalue_dx_local, Eigen::Matrix3d::Identity(), 1, h),
                 std::invalid_argument);
    const Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
    const Eigen::Vector4d gradient = Eigen::Vector4d::Ones();
    EXPECT_THROW(litepgo::add_gradient_wrt_dofs(dvalue_dx_local, gradient, std::numeric_limits<double>::infinity(), g),
                 std::invalid_argument);
    EXPECT_THROW(litepgo::add_hessian_wrt_dofs(dvalue_dx_local, H, std::numeric_limits<double>::quiet_NaN(), h),
                 std::invalid_argument);
    EXPECT_EQ(g, Eigen::Vector3d::Constant(7));
    EXPECT_EQ(h, Eigen::Matrix3d::Constant(9));
    g.resize(2);
    h.resize(3, 2);
    EXPECT_THROW(litepgo::add_gradient_wrt_dofs(dvalue_dx_local, gradient, 1, g), std::invalid_argument);
    EXPECT_THROW(litepgo::add_hessian_wrt_dofs(dvalue_dx_local, H, 1, h), std::invalid_argument);
}

TEST(DofDerivatives, AppliesWeightAfterPullbackToAvoidIntermediateOverflowAndUnderflow) {
    struct Scales {
        double derivative, material, weight, expected_gradient;
    };
    for (const auto scales : {Scales{1e-100, 1e200, 1e150, 1e250}, Scales{1e100, 1e-200, 1e-150, 1e-250}}) {
        Eigen::Matrix<double, 4, 3> dvalue_dx_local = Eigen::Matrix<double, 4, 3>::Zero();
        dvalue_dx_local.topRows<3>() = scales.derivative * Eigen::Matrix3d::Identity();
        const Eigen::Vector4d gradient = Eigen::Vector4d::Constant(scales.material);
        const Eigen::Matrix4d H = scales.material * Eigen::Matrix4d::Identity();
        Eigen::VectorXd g = Eigen::Vector3d::Zero();
        Eigen::MatrixXd h = Eigen::Matrix3d::Zero();
        litepgo::add_gradient_wrt_dofs(dvalue_dx_local, gradient, scales.weight, g);
        litepgo::add_hessian_wrt_dofs(dvalue_dx_local, H, scales.weight, h);
        EXPECT_TRUE(g.allFinite());
        EXPECT_TRUE(h.allFinite());
        EXPECT_TRUE((g / scales.expected_gradient).isApprox(Eigen::Vector3d::Ones(), 1e-13));
        EXPECT_TRUE((h / scales.weight).isIdentity(1e-13));
    }
}
