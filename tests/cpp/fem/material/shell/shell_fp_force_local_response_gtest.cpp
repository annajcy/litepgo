#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/shell_fp_force_local_response.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace litepgo::fem::test {
namespace {
using Local = ShellFpForceLocalResponse2D<KoiterStVK>;
Eigen::MatrixXd coordinates() {
    Eigen::MatrixXd X(6, 3);
    X << 0, 0, 0, 1.2, .1, .2, -.1, .9, -.1, 1.1, 1.3, .4, -1, .2, -.3, .2, -1.1, .3;
    return X;
}
struct PlasticField : Field2D {
    Eigen::VectorXd packed = Eigen::Matrix3d::Identity().reshaped();
    Eigen::MatrixXd D = Eigen::MatrixXd::Identity(9, 9);
    bool bad = false;
    mutable int calls = 0;
    int num_components() const override {
        return 9;
    }
    int num_dofs() const override {
        return static_cast<int>(D.cols());
    }
    Eigen::VectorXd eval_value(const Coordinate&) const override {
        return packed;
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Coordinate&) const override {
        ++calls;
        if (bad)
            return Eigen::MatrixXd::Constant(9, D.cols(), std::numeric_limits<double>::quiet_NaN());
        return D;
    }
};
TEST(ShellFpForceLocalResponse, FullJacobianMatchesForceAndEnergyDifferencesOnEveryStencilSize) {
    const Eigen::Vector2d xi(.2, .3), theta(2, 3);
    Eigen::Matrix3d P;
    P << 1.12, .08, -.03, .02, .91, .05, -.01, .04, 1.07;
    PlasticField field;
    for (int n : {3, 4, 5, 6}) {
        std::array<int, 3> opposite{-1, -1, -1};
        for (int k = 0; k < n - 3; ++k)
            opposite[k] = k + 3;
        const TriangleMidEdgeRestData rest(coordinates().topRows(n), opposite);
        Eigen::MatrixXd u = .03 * Eigen::MatrixXd::Random(n, 3);
        field.packed = Eigen::Matrix3d::Identity().reshaped();
        const Local local(rest, u, .2, field, theta);
        EXPECT_TRUE(local.eval_value(xi).isApprox(
            ShellMaterialLocalEnergy2D<KoiterStVK>(rest, u, .2, theta).eval_gradient(xi), 1e-12));
        field.packed = P.reshaped();
        const auto force = local.eval_value(xi);
        const auto J = local.eval_jacobian(xi);
        for (int j = 0; j < 9; ++j) {
            field.packed[j] += 1e-6;
            const auto plus = local.eval_value(xi);
            field.packed[j] -= 2e-6;
            const auto minus = local.eval_value(xi);
            field.packed[j] += 1e-6;
            EXPECT_LT((J.col(j) - (plus - minus) / 2e-6).norm(), 2e-8);
        }
        // Independent displacement derivative: x stays X+u while only the natural forms change.
        const Eigen::MatrixXd Y = rest.X_local * P.transpose();
        const auto natural = eval_triangle_mid_edge_kinematics(Y, opposite);
        auto density = [&](const Eigen::MatrixXd& displacement) {
            const auto current = eval_triangle_mid_edge_kinematics(rest.X_local + displacement, opposite);
            return KoiterStVK::eval_W({current.a, current.b, natural.a, natural.b, .2, theta});
        };
        for (int j = 0; j < 3 * n; ++j) {
            auto plus = u, minus = u;
            plus(j / 3, j % 3) += 1e-6;
            minus(j / 3, j % 3) -= 1e-6;
            EXPECT_NEAR(force[j], (density(plus) - density(minus)) / 2e-6, 2e-8);
        }
        Eigen::MatrixXd values = Eigen::MatrixXd::Constant(2, 3 * n, -5);
        Eigen::MatrixXd matrices = Eigen::MatrixXd::Constant(3 * n + 2, 11, -5);
        StridedVecRef value(values.row(1).transpose());
        StridedMatRef block(matrices.block(1, 1, 3 * n, 9));
        local.accumulate_into(xi, .4, value, &block);
        EXPECT_TRUE(value.isApprox((Eigen::VectorXd::Constant(3 * n, -5) + .4 * force).eval(), 1e-12));
        EXPECT_TRUE(block.isApprox((Eigen::MatrixXd::Constant(3 * n, 9, -5) + .4 * J).eval(), 1e-12));
        EXPECT_TRUE(values.row(0).isConstant(-5));
        EXPECT_TRUE(matrices.row(0).isConstant(-5));
        EXPECT_TRUE(matrices.col(10).isConstant(-5));
        const Eigen::Matrix3d R = Eigen::AngleAxisd(.4, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
        field.packed = (R * P).eval().reshaped();
        EXPECT_TRUE(local.eval_value(xi).isApprox(force, 1e-11));
    }
}
TEST(ShellFpForceLocalResponse, FieldChainRuleSnapshotsAndValidation) {
    static_assert(!std::is_constructible_v<Local, const TriangleMidEdgeRestData&, const Eigen::MatrixXd&, double,
                                           IdentityField2D&&, Eigen::VectorXd>);
    TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    const Eigen::MatrixXd u = .03 * Eigen::MatrixXd::Random(6, 3);
    const Eigen::Vector2d xi(.2, .3), theta(2, 3);
    PlasticField field;
    const Local full(rest, u, .2, field, theta);
    const auto J = full.eval_jacobian(xi);
    field.D = Eigen::MatrixXd::Random(9, 3);
    const Local reduced(rest, u, .2, field, theta);
    EXPECT_TRUE(reduced.eval_jacobian(xi).isApprox(J * field.D, 1e-12));
    const auto expected = reduced.eval_value(xi);
    rest.X_local.setZero();
    EXPECT_TRUE(reduced.eval_value(xi).isApprox(expected));
    field.calls = 0;
    field.bad = true;
    EXPECT_NO_THROW(reduced.eval_value(xi));
    EXPECT_EQ(field.calls, 0);
    EXPECT_THROW(reduced.eval_jacobian(xi), std::invalid_argument);
    field.bad = false;
    field.packed = Eigen::VectorXd::Zero(8);
    EXPECT_THROW(reduced.eval_value(xi), std::invalid_argument);
    field.packed = Eigen::VectorXd::Zero(9);
    EXPECT_THROW(reduced.eval_value(xi), std::domain_error);
    field.packed = Eigen::Matrix3d::Identity().reshaped();
    field.packed[0] = -1;
    EXPECT_THROW(reduced.eval_value(xi), std::domain_error);
    field.packed[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(reduced.eval_value(xi), std::invalid_argument);
    field.packed = Eigen::Matrix3d::Identity().reshaped();
    Vec value = Vec::Zero(17);
    EXPECT_THROW(reduced.accumulate(xi, 1, value), std::invalid_argument);
    value.resize(18);
    EXPECT_THROW(reduced.accumulate(xi, std::numeric_limits<double>::infinity(), value), std::invalid_argument);
    EXPECT_THROW(reduced.eval_value(Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
}
TEST(ShellFpForceLocalResponse, FlatStencilHasNormalInputNullspace) {
    Eigen::MatrixXd X = coordinates();
    X.col(2).setZero();
    const TriangleMidEdgeRestData rest(X, {3, 4, 5});
    PlasticField field;
    const Local local(rest, .02 * Eigen::MatrixXd::Random(6, 3), .2, field, Eigen::Vector2d(2, 3));
    const auto J = local.eval_jacobian(Eigen::Vector2d::Zero());
    EXPECT_TRUE(J.rightCols(3).isZero());
}
} // namespace
} // namespace litepgo::fem::test
