#include "../../../example/fem/hex_q1_mesh_energy.hpp"

#include <gtest/gtest.h>

#include <cmath>

TEST(HexQ1Example, GradientAndHessianMatchFiniteDifferences) {
    using namespace litepgo;
    const Vec X = example::hex_q1_rest_positions();
    auto energy = example::make_hex_q1_mesh_energy(X);
    ASSERT_EQ(X.size(), 36);
    for (int state : {0, 1, 2}) {
        SCOPED_TRACE(state);
        Vec x = 0.1 * X;
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
