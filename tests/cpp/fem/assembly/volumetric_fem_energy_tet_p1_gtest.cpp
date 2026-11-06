#include "../../testing/evaluate_energy.hpp"

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/integration/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/response/deformation_gradient_response.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/solver/newton.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace litepgo::fem::test {

inline Vec tet_p1_rest_positions() {
    // Two positively oriented tets share a face; their volumes are 1/6 and 1/3.
    Eigen::Matrix<double, 5, 3> rest_positions;
    rest_positions << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0.3, 0.2, -2;
    return rest_positions.reshaped<Eigen::RowMajor>();
}

inline auto make_tet_p1_energy(const Vec& rest_positions) {
    const mesh::TetTopology topology(5, {{0, 1, 2, 3}, {0, 2, 1, 4}});
    const dof::VertexDofMap rest_position_map(topology, 3);

    std::vector<std::shared_ptr<litepgo::energy::Response>> rest_position_fields;
    rest_position_fields.reserve(topology.num_cells());
    for (int e = 0; e < topology.num_cells(); ++e)
        rest_position_fields.push_back(
            std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(rest_position_map.gather(e, rest_positions)));
    auto Fp_field =
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(9, 3), Eigen::Matrix3d::Identity().reshaped());
    auto theta_field = std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(2, 3));
    return fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature>(
        rest_position_map, std::move(rest_position_fields),
        std::vector<std::shared_ptr<litepgo::energy::Response>>(topology.num_cells(), Fp_field),
        std::vector<std::shared_ptr<litepgo::energy::Response>>(topology.num_cells(), theta_field));
}

TEST(TetP1FemEnergy, GradientAndHessianMatchFiniteDifferences) {
    const Vec X_nodes = tet_p1_rest_positions();
    auto energy = make_tet_p1_energy(X_nodes);
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
        SpMat sparse_h;
        const double value = litepgo::testing::evaluate_energy(energy, x, &g, &sparse_h);
        const Eigen::MatrixXd H(sparse_h);
        ASSERT_EQ(g.size(), x.size());
        ASSERT_EQ(H.rows(), x.size());
        ASSERT_EQ(H.cols(), x.size());
        EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));

        // Direct edge-matrix evaluation checks the two element volumes and assembly.
        double expected = 0;
        for (const auto& cell : {mesh::TetTopology::Cell{0, 1, 2, 3}, mesh::TetTopology::Cell{0, 2, 1, 4}}) {
            Eigen::Matrix3d Dm, Du;
            for (int c = 0; c < 3; ++c) {
                Dm.col(c) =
                    X_nodes.segment<3>(3 * cell[static_cast<std::size_t>(c + 1)]) - X_nodes.segment<3>(3 * cell[0]);
                Du.col(c) = x.segment<3>(3 * cell[static_cast<std::size_t>(c + 1)]) - x.segment<3>(3 * cell[0]);
            }
            expected += Dm.determinant() / 6 *
                        fem::NeoHookean::eval_W({Eigen::Matrix3d::Identity() + Du * Dm.inverse(),
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
                fd_g[j] = (litepgo::testing::evaluate_energy(energy, plus, &plus_g, nullptr) -
                           litepgo::testing::evaluate_energy(energy, minus, &minus_g, nullptr)) /
                          (2 * h);
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

TEST(TetP1FemEnergy, ConstructionRejectsInvalidGeometry) {
    for (double scale : {0., -1., 1e200}) {
        SCOPED_TRACE(scale);
        const Vec X_nodes = scale * tet_p1_rest_positions();
        EXPECT_THROW(make_tet_p1_energy(X_nodes), std::invalid_argument);
    }
    EXPECT_THROW(make_tet_p1_energy(Vec::Zero(14)), std::invalid_argument);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        Vec X_nodes = tet_p1_rest_positions();
        X_nodes[0] = bad;
        EXPECT_THROW(make_tet_p1_energy(X_nodes), std::invalid_argument);
    }
}

TEST(TetP1FemEnergy, FeasibleStepBoundsTheWholeSegmentAndUsesBothElements) {
    const Vec X = tet_p1_rest_positions(), zero = Vec::Zero(X.size());
    auto energy = make_tet_p1_energy(X);
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

TEST(TetP1FemEnergy, NewtonRelaxesTheFreeVerticesWithFeasibleSteps) {
    auto energy = make_tet_p1_energy(tet_p1_rest_positions());
    Vec u = Vec::Zero(15);
    u[11] = 0.3;
    u[14] = -0.2;
    const double initial = energy.eval_value(u);
    const dof::DofReduction fixed(15, {0, 1, 2, 3, 4, 5, 6, 7, 8});
    solver::NewtonOptions options;
    options.gradient_tolerance = 1e-9;
    linalg::EigenLDLT solver;
    const auto result = solver::newton_solve(energy, fixed, u, options, solver);
    EXPECT_EQ(result.status, solver::NewtonStatus::Converged);
    EXPECT_GT(result.iterations, 0);
    EXPECT_LT(result.energy, initial);
    EXPECT_LT(std::abs(result.energy), 1e-12);
    EXPECT_TRUE(result.x.head(9).isZero(0));
}

TEST(TetP1FemEnergy, VolumetricFemEnergyRejectsIncompatibleMapsAndFields) {
    const mesh::TetTopology topology(5, {{0, 1, 2, 3}, {0, 2, 1, 4}});
    const dof::VertexDofMap rest_position_map(topology, 3), wrong_width(topology, 2);
    const dof::VertexDofMap wrong_count(mesh::TetTopology(5, {{0, 1, 2, 3}}), 3);
    const Vec rest_positions = tet_p1_rest_positions();

    const std::vector<std::shared_ptr<litepgo::energy::Response>> X{
        std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(rest_position_map.gather(0, rest_positions)),
        std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(rest_position_map.gather(1, rest_positions))};
    auto Fp =
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(9, 3), Eigen::Matrix3d::Identity().reshaped());
    auto theta = std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(2, 3));
    using Fem = fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature>;
    EXPECT_THROW((Fem(rest_position_map, {X[0]}, {Fp, Fp}, {theta, theta})), std::invalid_argument);
    EXPECT_THROW((Fem(rest_position_map,
                      {X[0], std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(Eigen::MatrixXd::Zero(4, 2))},
                      {Fp, Fp}, {theta, theta})),
                 std::invalid_argument);
    EXPECT_THROW((Fem(rest_position_map, X, {Fp}, {theta, theta})), std::invalid_argument);
    EXPECT_THROW((Fem(rest_position_map, X, {Fp, Fp}, {theta})), std::invalid_argument);
    EXPECT_THROW((Fem(wrong_width, X, {Fp, Fp}, {theta, theta})), std::invalid_argument);
    EXPECT_THROW((Fem(wrong_count, X, {Fp, Fp}, {theta, theta})), std::invalid_argument);
    EXPECT_THROW((Fem(rest_position_map, X, {Fp, theta}, {theta, theta})), std::invalid_argument);
}

TEST(TetP1FemEnergy, VolumetricFemEnergyAcceptsInterpolatedFixedFields) {
    const mesh::TetTopology topology(5, {{0, 1, 2, 3}, {0, 2, 1, 4}});
    const dof::VertexDofMap map(topology, 3);
    Eigen::MatrixXd plastic(4, 9), parameters(4, 2);
    for (int i = 0; i < 4; ++i) {
        Eigen::Matrix3d Fp = (1.0 + 0.01 * i) * Eigen::Matrix3d::Identity();
        plastic.row(i) = Fp.reshaped().transpose();
        parameters.row(i) = Eigen::RowVector2d(2 + i, 3 + i);
    }
    using Fem = fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature>;
    const Vec positions = tet_p1_rest_positions();

    const std::vector<std::shared_ptr<litepgo::energy::Response>> X{
        std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(map.gather(0, positions)),
        std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(map.gather(1, positions))};
    using InterpolatedFem = fem::VolumetricFemEnergy<fem::NeoHookean, fem::TetP1Basis, fem::TetCentroidQuadrature>;
    InterpolatedFem energy(map, X,
                           {std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(plastic),
                            std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(1.1 * plastic)},
                           {std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(parameters),
                            std::make_shared<fem::InterpolatedResponse<fem::TetP1Basis>>(2 * parameters)});
    const Eigen::VectorXd fp_mean = plastic.colwise().mean().transpose();
    const Eigen::VectorXd theta_mean = parameters.colwise().mean().transpose();
    Fem reference(
        map, X,
        {std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((fp_mean).size(), 3), fp_mean),
         std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((1.1 * fp_mean).size(), 3), 1.1 * fp_mean)},
        {std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((theta_mean).size(), 3), theta_mean),
         std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((2 * theta_mean).size(), 3),
                                                           2 * theta_mean)});
    const Vec u = 0.03 * positions;
    const dof::VertexDofMap first_map(mesh::TetTopology(5, {{0, 1, 2, 3}}), 3);
    const dof::VertexDofMap second_map(mesh::TetTopology(5, {{0, 2, 1, 4}}), 3);
    Fem first(first_map, {X[0]},
              {std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((fp_mean).size(), 3), fp_mean)},
              {std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((theta_mean).size(), 3), theta_mean)});
    Fem second(
        second_map, {X[1]},
        {std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((1.1 * fp_mean).size(), 3), 1.1 * fp_mean)},
        {std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((2 * theta_mean).size(), 3),
                                                           2 * theta_mean)});
    EXPECT_NEAR(energy.eval_value(u), first.eval_value(u) + second.eval_value(u), 1e-12);
    EXPECT_TRUE(energy.eval_gradient(u).isApprox(first.eval_gradient(u) + second.eval_gradient(u), 1e-12));
    EXPECT_TRUE(Eigen::MatrixXd(energy.eval_hessian(u))
                    .isApprox(Eigen::MatrixXd(first.eval_hessian(u)) + Eigen::MatrixXd(second.eval_hessian(u)), 1e-12));
    const Vec direction = -2 * positions;
    EXPECT_DOUBLE_EQ(energy.max_feasible_step(u, direction),
                     std::min(first.max_feasible_step(u, direction), second.max_feasible_step(u, direction)));
    EXPECT_NEAR(energy.eval_value(u), reference.eval_value(u), 1e-12);
    EXPECT_TRUE(energy.eval_gradient(u).isApprox(reference.eval_gradient(u), 1e-12));
    EXPECT_TRUE(Eigen::MatrixXd(energy.eval_hessian(u)).isApprox(Eigen::MatrixXd(reference.eval_hessian(u)), 1e-12));
    EXPECT_DOUBLE_EQ(energy.max_feasible_step(u, -2 * positions), reference.max_feasible_step(u, -2 * positions));
}

} // namespace litepgo::fem::test
