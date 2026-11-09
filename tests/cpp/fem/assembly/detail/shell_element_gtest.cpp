#include "../../testing/evaluate_shell_element.hpp"
#include "litepgo/fem/assembly/detail/shell_element.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace litepgo::fem::test {

namespace {

Eigen::MatrixXd coordinates() {
    Eigen::Matrix<double, 6, 3> X;
    X << 0, 0, 0, 1.2, 0.1, 0.2, -0.1, 0.9, -0.1, 1.1, 1.3, 0.4, -1, 0.2, -0.3, 0.2, -1.1, 0.3;
    return X;
}

Eigen::MatrixXd displacements(int vertices) {
    Eigen::MatrixXd u(vertices, 3);
    for (int v = 0; v < vertices; ++v)
        for (int c = 0; c < 3; ++c)
            u(v, c) = 0.03 * std::sin(0.7 * (3 * v + c + 1));
    return u;
}

struct NonFiniteGradientMaterial {
    static constexpr int num_parameters = KoiterStVK::num_parameters;
    static constexpr auto eval_W = KoiterStVK::eval_W;
    static constexpr auto eval_d2W_dab2 = KoiterStVK::eval_d2W_dab2;
    static Eigen::VectorXd eval_dW_dab(const ShellMaterialInputs&) {
        return Eigen::VectorXd::Constant(6, std::numeric_limits<double>::infinity());
    }
};
struct NonFiniteHessianMaterial {
    static constexpr int num_parameters = KoiterStVK::num_parameters;
    static constexpr auto eval_W = KoiterStVK::eval_W;
    static constexpr auto eval_dW_dab = KoiterStVK::eval_dW_dab;
    static Eigen::MatrixXd eval_d2W_dab2(const ShellMaterialInputs&) {
        return Eigen::MatrixXd::Constant(6, 6, std::numeric_limits<double>::quiet_NaN());
    }
};

// Owning results only for these tests; production uses pre-sized batch views.
template <class M = KoiterStVK>
auto evaluate(const TriangleMidEdgeRestData& rest, const Eigen::MatrixXd& u, int mask = 3, double thickness = 0.2,
              Vec theta = Eigen::Vector2d(2, 3)) {
    Vec g = Vec::Constant(u.size(), 123);
    Eigen::MatrixXd H = Eigen::MatrixXd::Constant(u.size(), u.size(), 123);
    StridedVecRef gv(g);
    StridedMatRef Hv(H);
    const double value =
        evaluate_shell_element<M>(rest, u, thickness, theta, mask & 1 ? &gv : nullptr, mask & 2 ? &Hv : nullptr);
    return std::make_tuple(value, g, H);
}

TEST(ShellElement, RejectsNonFiniteMaterialDerivatives) {
    const TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    const auto u = displacements(6);
    EXPECT_TRUE(std::isfinite(std::get<0>(evaluate<NonFiniteGradientMaterial>(rest, u, 0))));
    EXPECT_THROW(evaluate<NonFiniteGradientMaterial>(rest, u, 1), std::domain_error);
    EXPECT_THROW(evaluate<NonFiniteGradientMaterial>(rest, u, 2), std::domain_error);
    EXPECT_THROW(evaluate<NonFiniteHessianMaterial>(rest, u, 2), std::domain_error);
}
void check_derivatives(const TriangleMidEdgeRestData& rest) {
    const Eigen::MatrixXd u = displacements(static_cast<int>(rest.X_local.rows()));
    const auto [value, g, H] = evaluate(rest, u);
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
    constexpr double step = 2e-6;
    for (int j = 0; j < u.size(); ++j) {
        auto plus = u, minus = u;
        plus(j / 3, j % 3) += step;
        minus(j / 3, j % 3) -= step;
        const auto [vp, gp, Hp] = evaluate(rest, plus, 1);
        const auto [vm, gm, Hm] = evaluate(rest, minus, 1);
        EXPECT_NEAR(g[j], (vp - vm) / (2 * step), 2e-8);
        EXPECT_LT(((gp - gm) / (2 * step) - H.col(j)).norm(), 2e-7);
    }
    Eigen::MatrixXd J;
    const auto k = eval_triangle_mid_edge_kinematics(rest.X_local + u, rest.opposite, &J);
    const ShellMaterialInputs inputs{k.a, k.b, rest.a_bar, rest.b_bar, 0.2, Eigen::Vector2d(2, 3)};
    const Eigen::MatrixXd gn = 0.5 * rest.sqrt_det_a_bar * J.transpose() * KoiterStVK::eval_d2W_dab2(inputs) * J;
    EXPECT_GT((H - gn).norm(), 1e-5);
    for (int mask = 0; mask < 4; ++mask) {
        const auto [v, gm, Hm] = evaluate(rest, u, mask);
        EXPECT_DOUBLE_EQ(v, value);
        if (mask & 1)
            EXPECT_EQ(gm, g);
        else
            EXPECT_TRUE(gm.isConstant(123));
        if (mask & 2)
            EXPECT_EQ(Hm, H);
        else
            EXPECT_TRUE(Hm.isConstant(123));
    }
}
TEST(ShellElement, FullBoundaryAndDeduplicatedStencilsIncludeGeometricHessian) {
    check_derivatives(TriangleMidEdgeRestData(coordinates(), {3, 4, 5}));
    check_derivatives(TriangleMidEdgeRestData(coordinates().topRows(3), {-1, -1, -1}));
    Eigen::Matrix<double, 4, 3> X;
    X << 0, 0, 0, 1.2, 0, 0, 0, 0.9, 0, 0.1, 0.2, 1.1;
    check_derivatives(TriangleMidEdgeRestData(X, {3, 3, 3}));
}
TEST(ShellElement, StressFreeRigidStatesAndAreaAppliedExactlyOnce) {
    const auto X = coordinates();
    TriangleMidEdgeRestData rest(X, {3, 4, 5});
    const auto [v, g, H] = evaluate(rest, Eigen::MatrixXd::Zero(6, 3));
    EXPECT_DOUBLE_EQ(v, 0);
    EXPECT_TRUE(g.isZero(1e-13));
    EXPECT_GE(Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(H).eigenvalues().minCoeff(), -1e-11);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.4, Eigen::Vector3d(1, 2, -1).normalized()).toRotationMatrix();
    const Eigen::MatrixXd moved = (X * R.transpose()).rowwise() + Eigen::RowVector3d(0.3, -0.7, 1.1);
    const auto [vr, gr, Hr] = evaluate(rest, moved - X);
    EXPECT_NEAR(vr, 0, 1e-25);
    EXPECT_TRUE(gr.isZero(1e-12));
    const auto u = displacements(6);
    const auto [v1, g1, H1] = evaluate(rest, u);
    rest.sqrt_det_a_bar *= 2;
    const auto [v2, g2, H2] = evaluate(rest, u);
    EXPECT_DOUBLE_EQ(v2, 2 * v1);
    EXPECT_EQ(g2, 2 * g1);
    EXPECT_EQ(H2, 2 * H1);
}
TEST(ShellElement, RejectsMalformedInputsAndOutputShapesBeforeWriting) {
    const TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    for (const Eigen::MatrixXd& u : {Eigen::MatrixXd::Zero(3, 3).eval(), Eigen::MatrixXd::Zero(6, 2).eval()})
        EXPECT_THROW(evaluate(rest, u), std::invalid_argument);
    auto u = displacements(6);
    u(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(evaluate(rest, u), std::invalid_argument);
    for (double t : {0., -1., std::numeric_limits<double>::infinity()})
        EXPECT_THROW(evaluate(rest, displacements(6), 0, t), std::invalid_argument);
    for (int count : {0, 1, 3})
        EXPECT_THROW(evaluate(rest, displacements(6), 0, 0.2, Vec::Ones(count)), std::invalid_argument);
    Vec g = Vec::Ones(18);
    Eigen::MatrixXd H = Eigen::MatrixXd::Ones(17, 17);
    StridedVecRef gv(g);
    StridedMatRef Hv(H);
    EXPECT_THROW(evaluate_shell_element<KoiterStVK>(rest, displacements(6), 0.2, Eigen::Vector2d(2, 3), &gv, &Hv),
                 std::invalid_argument);
    EXPECT_TRUE(g.isOnes());
    EXPECT_TRUE(H.isOnes());
    g.resize(17);
    StridedVecRef wrong(g);
    EXPECT_THROW(evaluate_shell_element<KoiterStVK>(rest, displacements(6), 0.2, Eigen::Vector2d(2, 3), &wrong),
                 std::invalid_argument);
    auto bad = rest;
    bad.sqrt_det_a_bar = 0;
    EXPECT_THROW(evaluate(bad, displacements(6)), std::invalid_argument);
}
TEST(ShellElement, RejectsDegenerateGeometryAndInvalidMaterialDomain) {
    const TriangleMidEdgeRestData rest(coordinates(), {3, 4, 5});
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(6, 3);
    u.row(1) = rest.X_local.row(0) - rest.X_local.row(1);
    for (int mask = 0; mask < 4; ++mask)
        EXPECT_THROW(evaluate(rest, u, mask), std::domain_error);
    EXPECT_THROW(evaluate(rest, displacements(6), 0, 0.2, Eigen::Vector2d(-3, 2)), std::invalid_argument);
}
} // namespace
} // namespace litepgo::fem::test
