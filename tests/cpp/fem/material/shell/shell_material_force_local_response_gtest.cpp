#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/shell_material_force_local_response.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace litepgo::fem::test {
namespace {
using Local = ShellMaterialForceLocalResponse2D<KoiterStVK>;
struct Parameters : Field2D {
    Eigen::Vector3d coefficients{2, 3, .2};
    Eigen::Matrix<double, 2, 3> D;
    bool bad = false;
    mutable int derivative_calls = 0;
    Parameters() {
        D << 1, 0, .2, 0, 1, .3;
    }
    int num_components() const override {
        return 2;
    }
    int num_dofs() const override {
        return 3;
    }
    Eigen::VectorXd eval_value(const Coordinate& xi) const override {
        return (1 + xi.x()) * D * coefficients;
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Coordinate& xi) const override {
        ++derivative_calls;
        if (bad)
            return Eigen::MatrixXd::Constant(2, 3, std::numeric_limits<double>::quiet_NaN());
        return (1 + xi.x()) * D;
    }
};
Eigen::MatrixXd positions() {
    Eigen::MatrixXd X(6, 3);
    X << 0, 0, 0, 1.2, .1, .2, -.1, .9, -.1, 1.1, 1.3, .4, -1, .2, -.3, .2, -1.1, .3;
    return X;
}
TEST(ShellMaterialForceLocalResponse, MatchesEnergyAndRectangularParameterFiniteDifferences) {
    Parameters theta;
    const Eigen::Vector2d xi(.2, .3);
    for (int n : {3, 4, 5, 6}) {
        std::array<int, 3> opposite{-1, -1, -1};
        for (int j = 0; j < n - 3; ++j)
            opposite[j] = j + 3;
        const TriangleMidEdgeRestData rest(positions().topRows(n), opposite);
        Eigen::MatrixXd u = .03 * Eigen::MatrixXd::Random(n, 3);
        Local local(rest, u, .2, theta);
        const auto expected =
            ShellMaterialLocalEnergy2D<KoiterStVK>(rest, u, .2, theta.eval_value(xi)).eval_gradient(xi);
        EXPECT_TRUE(local.eval_value(xi).isApprox(expected, 1e-12));
        const auto J = local.eval_jacobian(xi);
        EXPECT_EQ(J.rows(), 3 * n);
        EXPECT_EQ(J.cols(), 3);
        for (int j = 0; j < 3; ++j) {
            theta.coefficients[j] += 1e-6;
            const auto plus = local.eval_value(xi);
            theta.coefficients[j] -= 2e-6;
            const auto minus = local.eval_value(xi);
            theta.coefficients[j] += 1e-6;
            EXPECT_LT((J.col(j) - (plus - minus) / 2e-6).norm(), 1e-8);
        }
        Eigen::MatrixXd values = Eigen::MatrixXd::Constant(2, 3 * n, -5);
        Eigen::MatrixXd matrices = Eigen::MatrixXd::Constant(3 * n + 2, 5, -5);
        StridedVecRef value(values.row(1).transpose());
        StridedMatRef block(matrices.block(1, 1, 3 * n, 3));
        local.accumulate_into(xi, .4, value, &block);
        EXPECT_TRUE(value.isApprox((Eigen::VectorXd::Constant(3 * n, -5) + .4 * expected).eval(), 1e-12));
        EXPECT_TRUE(block.isApprox((Eigen::MatrixXd::Constant(3 * n, 3, -5) + .4 * J).eval(), 1e-12));
        EXPECT_TRUE(values.row(0).isConstant(-5));
        EXPECT_TRUE(matrices.row(0).isConstant(-5));
        EXPECT_TRUE(matrices.col(4).isConstant(-5));
        u.setZero();
        EXPECT_TRUE(local.eval_value(xi).isApprox(expected, 1e-12));
    }
}
TEST(ShellMaterialForceLocalResponse, ValidatesInputsAndOnlyRequestsParameterDerivativesWhenNeeded) {
    static_assert(!std::is_constructible_v<Local, const TriangleMidEdgeRestData&, const Eigen::MatrixXd&, double,
                                           IdentityField2D&&>);
    const TriangleMidEdgeRestData rest(positions().topRows(3), {-1, -1, -1});
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(3, 3);
    Parameters theta;
    Local local(rest, u, .2, theta);
    theta.bad = true;
    EXPECT_NO_THROW(local.eval_value(Eigen::Vector2d::Zero()));
    EXPECT_EQ(theta.derivative_calls, 0);
    EXPECT_THROW(local.eval_jacobian(Eigen::Vector2d::Zero()), std::invalid_argument);
    EXPECT_THROW((Local(rest, u, 0, theta)), std::invalid_argument);
    EXPECT_THROW((Local(rest, Eigen::MatrixXd::Zero(3, 2), .2, theta)), std::invalid_argument);
    auto bad_rest = rest;
    bad_rest.X_local.resize(3, 2);
    EXPECT_THROW((Local(bad_rest, u, .2, theta)), std::invalid_argument);
    IdentityField2D wrong(Eigen::Vector3d::Ones());
    EXPECT_THROW((Local(rest, u, .2, wrong)), std::invalid_argument);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(9);
    EXPECT_THROW(local.accumulate(Eigen::Vector2d::Zero(), std::numeric_limits<double>::infinity(), v),
                 std::invalid_argument);
    EXPECT_THROW(local.eval_value(Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
    v.resize(8);
    EXPECT_THROW(local.accumulate(Eigen::Vector2d::Zero(), 1, v), std::invalid_argument);
    theta.coefficients << 1, -2, 0;
    EXPECT_THROW(local.eval_value(Eigen::Vector2d::Zero()), std::invalid_argument);
}
} // namespace
} // namespace litepgo::fem::test
