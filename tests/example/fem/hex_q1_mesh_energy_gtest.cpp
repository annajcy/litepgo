#include "../../../example/fem/hex_q1_mesh_energy.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

TEST(HexQ1Example, GradientAndHessianMatchFiniteDifferences) {
    using namespace litepgo;
    const Vec X_nodes = example::hex_q1_rest_positions();
    auto energy = example::make_hex_q1_mesh_energy(X_nodes);
    ASSERT_EQ(X_nodes.size(), 36);
    for (int state : {0, 1, 2}) {
        SCOPED_TRACE(state);
        Vec x = 0.1 * X_nodes;
        if (state == 0)
            x.setZero();
        if (state == 2)
            for (int j = 0; j < x.size(); ++j)
                x[j] += 0.02 * std::sin(j + 0.3);
        Vec g;
        SpMat sparse_h = energy.hessian_pattern();
        energy.evaluate(x, &g, &sparse_h);
        const Eigen::MatrixXd H(sparse_h);
        ASSERT_EQ(g.size(), x.size());
        ASSERT_EQ(H.rows(), x.size());
        ASSERT_EQ(H.cols(), x.size());
        EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
        for (double h : {1e-3, 3e-4, 1e-4}) {
            SCOPED_TRACE(h);
            Vec fd_g(x.size());
            Eigen::MatrixXd fd_h(x.size(), x.size());
            for (int j = 0; j < x.size(); ++j) {
                Vec plus = x, minus = x, plus_g, minus_g;
                plus[j] += h;
                minus[j] -= h;
                fd_g[j] =
                    (energy.evaluate(plus, &plus_g, nullptr) - energy.evaluate(minus, &minus_g, nullptr)) / (2 * h);
                fd_h.col(j) = (plus_g - minus_g) / (2 * h);
            }
            EXPECT_LT((g - fd_g).norm() / (1 + g.norm()), 10 * h * h + 1e-9);
            EXPECT_LT((H - fd_h).norm() / (1 + H.norm()), 10 * h * h + 1e-9);

            // Check curvature against scalar energy, independently of the analytic gradient.
            Vec d(x.size()), v(x.size());
            for (int j = 0; j < x.size(); ++j) {
                d[j] = std::sin(j + 0.5);
                v[j] = std::cos(j + 0.2);
            }
            d.normalize();
            v.normalize();
            const double mixed = (energy.value(x + h * d + h * v) - energy.value(x + h * d - h * v) -
                                  energy.value(x - h * d + h * v) + energy.value(x - h * d - h * v)) /
                                 (4 * h * h);
            EXPECT_NEAR(d.dot(H * v), mixed, (10 * h * h + 1e-7) * (1 + H.norm()));
        }
    }
}

TEST(HexQ1Example, ElementLambdaRejectsInvalidGeometry) {
    using namespace litepgo;
    for (double scale : {0., -1., 1e200}) {
        SCOPED_TRACE(scale);
        const Vec X_nodes = scale * example::hex_q1_rest_positions();
        auto energy = example::make_hex_q1_mesh_energy(X_nodes);
        const Vec x = Vec::Zero(36);
        Vec g;
        SpMat H = energy.hessian_pattern();
        EXPECT_THROW(energy.value(x), std::invalid_argument);
        EXPECT_THROW(energy.evaluate(x, &g, nullptr), std::invalid_argument);
        EXPECT_THROW(energy.evaluate(x, nullptr, &H), std::invalid_argument);
    }
    EXPECT_THROW(example::make_hex_q1_mesh_energy(Vec::Zero(35)), std::invalid_argument);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        Vec X_nodes = example::hex_q1_rest_positions();
        X_nodes[0] = bad;
        EXPECT_THROW(example::make_hex_q1_mesh_energy(X_nodes), std::invalid_argument);
    }
}

TEST(HexQ1Example, ElementLambdaIntegratesTranslatedWarpedGeometryAndIndependentOutputs) {
    using namespace litepgo;
    Vec X_nodes = example::hex_q1_rest_positions();
    for (int v = 0; v < 12; ++v) {
        const Eigen::Vector3d X = X_nodes.segment<3>(3 * v);
        X_nodes.segment<3>(3 * v) = Eigen::Vector3d(X.x() + 0.4 * X.y() * X.z(), X.y() + 0.2 * X.x() * X.z(), X.z()) +
                                    Eigen::Vector3d(1, -2, 3);
    }
    auto energy = example::make_hex_q1_mesh_energy(X_nodes);
    const Vec x = 0.1 * X_nodes;
    Vec g = Vec::Ones(1);
    SpMat H = energy.hessian_pattern();
    const double value = energy.evaluate(x, &g, &H);
    // det(dX/dxi) = cell_width * (1 - 0.08*z^2); integrate over both cells.
    const double volume = 3 * (1 - 0.08 / 3);
    EXPECT_NEAR(value,
                volume * NeoHookean::value(
                             {1.1 * Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(), Eigen::Vector2d(2, 3)}),
                1e-13);
    Vec g_only = Vec::Constant(2, 7);
    SpMat H_only = energy.hessian_pattern();
    EXPECT_DOUBLE_EQ(energy.value(x), value);
    EXPECT_DOUBLE_EQ(energy.evaluate(x, &g_only, nullptr), value);
    EXPECT_DOUBLE_EQ(energy.evaluate(x, nullptr, &H_only), value);
    EXPECT_TRUE(g_only.isApprox(g, 1e-13));
    EXPECT_TRUE(Eigen::MatrixXd(H_only).isApprox(Eigen::MatrixXd(H), 1e-13));
    EXPECT_DOUBLE_EQ(energy.evaluate(x, &g, &H), value);
    EXPECT_TRUE(g.isApprox(g_only, 1e-13));
    EXPECT_TRUE(Eigen::MatrixXd(H).isApprox(Eigen::MatrixXd(H_only), 1e-13));
}

TEST(HexQ1Example, ElementLambdaRejectsGeometryInvertedAtAQuadraturePoint) {
    using namespace litepgo;
    Vec X_nodes = example::hex_q1_rest_positions();
    for (int v = 0; v < 12; ++v)
        X_nodes[3 * v + 2] *= 1 - 1.5 * X_nodes[3 * v];
    auto energy = example::make_hex_q1_mesh_energy(X_nodes);
    EXPECT_THROW(energy.value(Vec::Zero(36)), std::invalid_argument);
}
