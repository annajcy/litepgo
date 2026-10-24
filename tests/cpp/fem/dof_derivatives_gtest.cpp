#include "litepgo/fem/detail/dof_derivatives.hpp"

#include <gtest/gtest.h>

namespace {
template <class J, class G>
concept CanConvertGradient = requires(const J& j, const G& g) { litepgo::detail::gradient_wrt_dofs(j, g); };
template <class J, class H>
concept CanConvertHessian = requires(const J& j, const H& h) { litepgo::detail::hessian_wrt_dofs(j, h); };
static_assert(CanConvertGradient<Eigen::Matrix<double, 4, 3>, Eigen::Matrix2d>);
static_assert(!CanConvertGradient<Eigen::Matrix3d, Eigen::Matrix2d>);
static_assert(!CanConvertHessian<Eigen::Matrix<double, 4, 3>, Eigen::Matrix3d>);
static_assert(!CanConvertGradient<Eigen::MatrixXd, Eigen::VectorXd>);

TEST(DofDerivatives, AffineChainRuleMatchesFiniteDifferences) {
    Eigen::Matrix<double, 4, 3> J;
    J << 1, 0.2, -0.3, 0.1, 2, 0.4, -0.2, 0.3, 0.8, 0.7, -0.4, 1.2;
    Eigen::Matrix4d H = Eigen::Vector4d(1, 2, 3, 4).asDiagonal();
    H(0, 3) = H(3, 0) = 0.2;
    const Eigen::Vector3d a(0.2, -0.3, 0.5);
    const Eigen::Vector4d offset(0.3, 0.1, -0.2, 0.4);
    const auto energy = [&](const Eigen::Vector3d& x) {
        const Eigen::Vector4d y = J * x + offset;
        return 0.5 * y.dot(H * y);
    };
    const auto gradient = [&](const Eigen::Vector3d& x) {
        const Eigen::Vector4d g = H * (J * x + offset);
        const Eigen::Matrix<double, 2, 2, Eigen::RowMajor> matrix = Eigen::Map<const Eigen::Matrix2d>(g.data());
        return litepgo::detail::gradient_wrt_dofs(J, matrix);
    };
    const auto g = gradient(a);
    const auto hessian = litepgo::detail::hessian_wrt_dofs(J, H);
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
    const auto copied = litepgo::detail::hessian_wrt_dofs(2 * J, H + H);
    EXPECT_TRUE(copied.isApprox(8 * hessian, 1e-13));
}
} // namespace
