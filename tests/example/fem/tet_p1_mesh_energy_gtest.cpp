#include "../../../example/fem/tet_p1_mesh_energy.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/solver/newton.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

#include <cmath>
#include <limits>

TEST(TetP1Example, GradientAndHessianMatchFiniteDifferences) {
    using namespace litepgo;
    const Vec X_nodes = example::tet_p1_rest_positions();
    auto energy = example::make_tet_p1_mesh_energy(X_nodes);
    ASSERT_EQ(X_nodes.size(), 15);
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
        const double value = energy.eval(x, &g, &sparse_h);
        const Eigen::MatrixXd H(sparse_h);
        ASSERT_EQ(g.size(), x.size());
        ASSERT_EQ(H.rows(), x.size());
        ASSERT_EQ(H.cols(), x.size());
        EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));

        // Direct edge-matrix evaluation checks the two element volumes and assembly.
        double expected = 0;
        for (const auto& cell : {TetTopology::Cell{0, 1, 2, 3}, TetTopology::Cell{0, 2, 1, 4}}) {
            Eigen::Matrix3d Dm, Du;
            for (int c = 0; c < 3; ++c) {
                Dm.col(c) =
                    X_nodes.segment<3>(3 * cell[static_cast<std::size_t>(c + 1)]) - X_nodes.segment<3>(3 * cell[0]);
                Du.col(c) = x.segment<3>(3 * cell[static_cast<std::size_t>(c + 1)]) - x.segment<3>(3 * cell[0]);
            }
            expected += Dm.determinant() / 6 *
                        NeoHookean::eval_W({Eigen::Matrix3d::Identity() + Du * Dm.inverse(),
                                            Eigen::Matrix3d::Identity(), Eigen::Vector2d(2, 3)});
        }
        EXPECT_NEAR(value, expected, 1e-13);
        for (double h : {1e-3, 3e-4, 1e-4}) {
            SCOPED_TRACE(h);
            Vec fd_g(x.size());
            Eigen::MatrixXd fd_h(x.size(), x.size());
            for (int j = 0; j < x.size(); ++j) {
                Vec plus = x, minus = x, plus_g, minus_g;
                plus[j] += h;
                minus[j] -= h;
                fd_g[j] = (energy.eval(plus, &plus_g, nullptr) - energy.eval(minus, &minus_g, nullptr)) / (2 * h);
                fd_h.col(j) = (plus_g - minus_g) / (2 * h);
            }
            EXPECT_LT((g - fd_g).norm() / (1 + g.norm()), 10 * h * h + 1e-9);
            EXPECT_LT((H - fd_h).norm() / (1 + H.norm()), 10 * h * h + 1e-9);

            Vec d(x.size()), v(x.size());
            for (int j = 0; j < x.size(); ++j) {
                d[j] = std::sin(j + 0.5);
                v[j] = std::cos(j + 0.2);
            }
            d.normalize();
            v.normalize();
            const double mixed = (energy.eval_value(x + h * d + h * v) - energy.eval_value(x + h * d - h * v) -
                                  energy.eval_value(x - h * d + h * v) + energy.eval_value(x - h * d - h * v)) /
                                 (4 * h * h);
            EXPECT_NEAR(d.dot(H * v), mixed, (10 * h * h + 1e-7) * (1 + H.norm()));
        }
    }
}

TEST(TetP1Example, ConstructionRejectsInvalidGeometry) {
    using namespace litepgo;
    for (double scale : {0., -1., 1e200}) {
        SCOPED_TRACE(scale);
        const Vec X_nodes = scale * example::tet_p1_rest_positions();
        EXPECT_THROW(example::make_tet_p1_mesh_energy(X_nodes), std::invalid_argument);
    }
    EXPECT_THROW(example::make_tet_p1_mesh_energy(Vec::Zero(14)), std::invalid_argument);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        Vec X_nodes = example::tet_p1_rest_positions();
        X_nodes[0] = bad;
        EXPECT_THROW(example::make_tet_p1_mesh_energy(X_nodes), std::invalid_argument);
    }
}

TEST(TetP1Example, FeasibleStepBoundsTheWholeSegmentAndUsesBothElements) {
    using namespace litepgo;
    const Vec X = example::tet_p1_rest_positions(), zero = Vec::Zero(X.size());
    auto energy = example::make_tet_p1_mesh_energy(X);
    EXPECT_EQ(energy.max_feasible_step(zero, zero), 1);
    EXPECT_EQ(energy.max_feasible_step(zero, 0.1 * X), 1);
    Vec direction = zero;
    for (int v = 0; v < X.size() / 3; ++v) {
        direction[3 * v] = -4 * X[3 * v];
        direction[3 * v + 1] = -4 * X[3 * v + 1];
    }
    EXPECT_TRUE(std::isfinite(energy.eval_value(direction)));
    const double alpha = energy.max_feasible_step(zero, direction);
    EXPECT_NEAR(alpha, 0.225, 1e-6);
    for (int i = 0; i <= 20; ++i)
        EXPECT_TRUE(std::isfinite(energy.eval_value((alpha * i / 20) * direction)));
    direction.setZero();
    direction[14] = 16;
    EXPECT_NEAR(energy.max_feasible_step(zero, direction), 0.1125, 1e-12);
    EXPECT_THROW(energy.max_feasible_step(-X, zero), std::domain_error);
}

TEST(TetP1Example, NewtonRelaxesTheFreeVerticesWithFeasibleSteps) {
    using namespace litepgo;
    auto energy = example::make_tet_p1_mesh_energy(example::tet_p1_rest_positions());
    Vec u = Vec::Zero(15);
    u[11] = 0.3;
    u[14] = -0.2;
    const double initial = energy.eval_value(u);
    const DofReduction fixed(15, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    NewtonOptions options;
    options.gradient_tolerance = 1e-9;
    EigenLDLT solver;
    const auto result = newton_solve(energy, fixed, u, options, solver);
    EXPECT_EQ(result.status, NewtonStatus::Converged);
    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.energy, initial);
    EXPECT_LT(std::abs(result.energy), 1e-12);
    EXPECT_TRUE(result.x.head(9).isZero(0));
}
