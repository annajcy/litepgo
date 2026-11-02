
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"

#include <memory>
#include <utility>
#include <vector>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace litepgo::example::test {

inline Vec hex_q1_rest_positions() {
    // Two adjacent hexes of widths 1 and 2, sharing the face at X = 1.
    Eigen::Matrix<double, 12, 3> rest_positions;
    rest_positions << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 3, 0, 0, 3, 1, 0, 3, 0, 1,
        3, 1, 1;
    return rest_positions.reshaped<Eigen::RowMajor>();
}

inline auto make_hex_q1_corner_pull(const Vec& rest_positions) {
    const mesh::HexTopology topology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
    const dof::VertexDofMap rest_position_map(topology, 3);
    using XField = fem::InterpolatedField3D<fem::HexQ1Basis>;
    std::vector<XField> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.emplace_back(rest_position_map.gather(e, rest_positions));
    const fem::IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const fem::IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    return fem::VolumetricFemEnergy<fem::NeoHookean, fem::HexQ1Basis, fem::GaussLegendre3D<2>, XField,
                                    fem::IdentityField3D, fem::IdentityField3D>(
        rest_position_map, std::move(rest_position_fields),
        std::vector<fem::IdentityField3D>(topology.num_cells(), Fp_field),
        std::vector<fem::IdentityField3D>(topology.num_cells(), theta_field));
}

TEST(HexQ1Example, GradientAndHessianMatchFiniteDifferences) {
    const Vec X_nodes = hex_q1_rest_positions();
    auto energy = make_hex_q1_corner_pull(X_nodes);
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
        energy.eval(x, &g, &sparse_h);
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
                fd_g[j] = (energy.eval(plus, &plus_g, nullptr) - energy.eval(minus, &minus_g, nullptr)) / (2 * h);
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
            const double mixed = (energy.eval_value(x + h * d + h * v) - energy.eval_value(x + h * d - h * v) -
                                  energy.eval_value(x - h * d + h * v) + energy.eval_value(x - h * d - h * v)) /
                                 (4 * h * h);
            EXPECT_NEAR(d.dot(H * v), mixed, (10 * h * h + 1e-7) * (1 + H.norm()));
        }
    }
}

TEST(HexQ1Example, ConstructionRejectsInvalidGeometry) {
    for (double scale : {0., -1., 1e200}) {
        SCOPED_TRACE(scale);
        const Vec X_nodes = scale * hex_q1_rest_positions();
        EXPECT_THROW(make_hex_q1_corner_pull(X_nodes), std::invalid_argument);
    }
    EXPECT_THROW(make_hex_q1_corner_pull(Vec::Zero(35)), std::invalid_argument);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        Vec X_nodes = hex_q1_rest_positions();
        X_nodes[0] = bad;
        EXPECT_THROW(make_hex_q1_corner_pull(X_nodes), std::invalid_argument);
    }
}

TEST(HexQ1Example, CachedElementIntegratesTranslatedWarpedGeometryAndIndependentOutputs) {
    Vec X_nodes = hex_q1_rest_positions();
    for (int v = 0; v < 12; ++v) {
        const Eigen::Vector3d X = X_nodes.segment<3>(3 * v);
        X_nodes.segment<3>(3 * v) = Eigen::Vector3d(X.x() + 0.4 * X.y() * X.z(), X.y() + 0.2 * X.x() * X.z(), X.z()) +
                                    Eigen::Vector3d(1, -2, 3);
    }
    auto energy = make_hex_q1_corner_pull(X_nodes);
    const Vec x = 0.1 * X_nodes;
    Vec g = Vec::Ones(1);
    SpMat H = energy.hessian_pattern();
    const double value = energy.eval(x, &g, &H);
    // det(dX/dxi) = cell_width * (1 - 0.08*z^2); integrate over both cells.
    const double volume = 3 * (1 - 0.08 / 3);
    EXPECT_NEAR(value,
                volume * fem::NeoHookean::eval_W(
                             {1.1 * Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(), Eigen::Vector2d(2, 3)}),
                1e-13);
    Vec g_only = Vec::Constant(2, 7);
    SpMat H_only = energy.hessian_pattern();
    EXPECT_DOUBLE_EQ(energy.eval_value(x), value);
    EXPECT_DOUBLE_EQ(energy.eval(x, &g_only, nullptr), value);
    EXPECT_DOUBLE_EQ(energy.eval(x, nullptr, &H_only), value);
    EXPECT_TRUE(g_only.isApprox(g, 1e-13));
    EXPECT_TRUE(Eigen::MatrixXd(H_only).isApprox(Eigen::MatrixXd(H), 1e-13));
    EXPECT_DOUBLE_EQ(energy.eval(x, &g, &H), value);
    EXPECT_TRUE(g.isApprox(g_only, 1e-13));
    EXPECT_TRUE(Eigen::MatrixXd(H).isApprox(Eigen::MatrixXd(H_only), 1e-13));
}

TEST(HexQ1Example, ConstructionRejectsGeometryInvertedAtAQuadraturePoint) {
    Vec X_nodes = hex_q1_rest_positions();
    for (int v = 0; v < 12; ++v)
        X_nodes[3 * v + 2] *= 1 - 1.5 * X_nodes[3 * v];
    EXPECT_THROW(make_hex_q1_corner_pull(X_nodes), std::invalid_argument);
}

TEST(HexQ1Example, FeasibleStepDetectsInteriorCollapseAndTheMostRestrictiveElement) {
    const Vec X = hex_q1_rest_positions(), zero = Vec::Zero(X.size());
    auto energy = make_hex_q1_corner_pull(X);
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
    for (int v = 0; v < X.size() / 3; ++v)
        direction[3 * v] = -4 * std::max(0.0, X[3 * v] - 1);
    EXPECT_NEAR(energy.max_feasible_step(zero, direction), 0.225, 1e-12);
    EXPECT_THROW(energy.max_feasible_step(-X, zero), std::domain_error);
}

} // namespace litepgo::example::test
