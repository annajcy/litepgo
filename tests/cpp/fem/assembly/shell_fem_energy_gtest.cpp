#include "../../testing/finite_difference.hpp"
#include "../testing/evaluate_shell_element.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/assembly/shell_fem_force_fp_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/pointwise_shell_adapter.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace litepgo::fem::test {

namespace {
mesh::TriangleTopology topology() {
    // Face 0 has three neighbors; boundary faces each have one neighbor.
    return mesh::TriangleTopology(6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}});
}
Vec rest_positions() {
    // A flat patch with face areas 1, 2, 1/2, 2 and consistent winding.
    Eigen::Matrix<double, 6, 3> X_nodes;
    X_nodes << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, -1, 0, 0, 0, -2, 0;
    return X_nodes.reshaped<Eigen::RowMajor>();
}
struct Patch {
    dof::StencilDofMap map{6, mesh::make_triangle_mid_edge_stencils(topology()).vertices, 3};
    std::vector<TriangleMidEdgeRestData> rest;
    std::vector<double> thickness{0.1, 0.2, 0.15, 0.3};
    std::vector<Vec> theta{Eigen::Vector2d(2, 3), Eigen::Vector2d(4, 5), Eigen::Vector2d(6, 7), Eigen::Vector2d(8, 9)};
    Patch() {
        const auto stencils = mesh::make_triangle_mid_edge_stencils(topology());
        const Vec X = rest_positions();
        for (int e = 0; e < 4; ++e)
            rest.emplace_back(map.gather(e, X), stencils.opposite[static_cast<std::size_t>(e)]);
    }
};

TEST(ShellFemEnergy, PerFaceInputsMatchExplicitIntegrationAndScatter) {
    Patch patch;
    ShellFemEnergy energy(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), patch.map, patch.rest,
                          patch.thickness, patch.theta);
    Vec u = Vec::Zero(18);
    u[11] = 0.2;
    u[14] = -0.15;
    u[17] = 0.18;
    const Vec du = -2 * rest_positions();
    double expected = 0, step = 1;
    Vec g = Vec::Zero(18);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(18, 18);
    for (int e = 0; e < 4; ++e) {
        const auto index = static_cast<std::size_t>(e);
        const auto& rest = patch.rest[index];
        const auto local_u = patch.map.gather(e, u);
        Vec local_g(local_u.size());
        Eigen::MatrixXd local_H(local_u.size(), local_u.size());
        StridedVecRef gradient(local_g);
        StridedMatRef hessian(local_H);
        expected += evaluate_shell_element<KoiterStVK>(rest, local_u, patch.thickness[index], patch.theta[index],
                                                       &gradient, &hessian);
        const auto indices = patch.map.dof_indices(e);
        for (std::size_t i = 0; i < indices.size(); ++i) {
            g[indices[i]] += local_g[static_cast<Eigen::Index>(i)];
            for (std::size_t j = 0; j < indices.size(); ++j)
                H(indices[i], indices[j]) += local_H(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
        step = std::min(
            step, eval_triangle_mid_edge_feasible_step(rest.X_local + local_u, patch.map.gather(e, du), rest.opposite));
    }
    EXPECT_NEAR(energy.eval_value(u), expected, 1e-12);
    EXPECT_TRUE(energy.eval_gradient(u).isApprox(g, 1e-12));
    EXPECT_TRUE(Eigen::MatrixXd(energy.eval_hessian(u)).isApprox(H, 1e-12));
    EXPECT_DOUBLE_EQ(energy.max_feasible_step(u, du), step);
    energy::test::expect_derivatives_match(energy, u, 2e-6);
    patch.rest[0].X_local.setZero();
    patch.thickness[0] = 100;
    patch.theta[0].setZero();
    EXPECT_NEAR(energy.eval_value(u), expected, 1e-12);
}

TEST(ShellFemEnergy, RejectsInvalidArraysStencilsThicknessAndMaterial) {
    Patch p;
    using Shell = ShellFemEnergy;
    EXPECT_THROW(
        (Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, {}, p.thickness, p.theta)),
        std::invalid_argument);
    EXPECT_THROW(
        (Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest, {}, p.theta)),
        std::invalid_argument);
    EXPECT_THROW(
        (Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest, p.thickness, {})),
        std::invalid_argument);
    const auto vertices = mesh::make_triangle_mid_edge_stencils(topology()).vertices;
    EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(),
                        dof::StencilDofMap(6, vertices, 2), p.rest, p.thickness, p.theta)),
                 std::invalid_argument);
    auto wrong = vertices;
    wrong[0].pop_back();
    EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(),
                        dof::StencilDofMap(6, wrong, 3), p.rest, p.thickness, p.theta)),
                 std::invalid_argument);
    for (double value : {0., -1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        auto thickness = p.thickness;
        thickness[3] = value;
        EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest,
                            thickness, p.theta)),
                     std::invalid_argument);
        auto rest = p.rest;
        rest[3].sqrt_det_a_bar = value;
        EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, rest,
                            p.thickness, p.theta)),
                     std::invalid_argument);
    }
    for (const Vec& value : {Vec::Ones(3).eval(), Vec(Eigen::Vector2d(2, -1)),
                             Vec::Constant(2, std::numeric_limits<double>::quiet_NaN()).eval()}) {
        auto theta = p.theta;
        theta[2] = value;
        EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest,
                            p.thickness, theta)),
                     std::invalid_argument);
    }
    p.rest[0].opposite[0] = 100;
    EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest, p.thickness,
                        p.theta)),
                 std::invalid_argument);
}

TEST(ShellFemEnergy, EmptyMeshProducesZeroAndFullStep) {
    ShellFemEnergy energy(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(),
                          dof::StencilDofMap(0, {}, 3), {}, {}, {});
    EXPECT_EQ(energy.eval_value(Vec(0)), 0);
    EXPECT_EQ(energy.eval_gradient(Vec(0)).size(), 0);
    EXPECT_EQ(energy.eval_hessian(Vec(0)).rows(), 0);
    EXPECT_EQ(energy.max_feasible_step(Vec(0), Vec(0)), 1);
}

TEST(ShellFemEnergy, PlasticFormsPreserveOriginalMeasureAndMatchResponse) {
    Patch p;
    // Curved reference exercises both transformed natural metric and curvature.
    Vec X = rest_positions();
    X[11] = .2;
    X[14] = -.1;
    X[17] = .15;
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology());
    for (int e = 0; e < 4; ++e)
        p.rest[e] = TriangleMidEdgeRestData(p.map.gather(e, X), stencils.opposite[e]);
    Eigen::Matrix3d P;
    P << 1.12, .08, -.03, .02, .94, .04, -.01, .03, 1.05;
    std::vector<Eigen::Matrix3d> plastic(4, P);
    Vec u = .02 * Vec::LinSpaced(18, -1, 1);
    ShellFemEnergy elastic(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest,
                           p.thickness, p.theta, plastic);
    ShellFemEnergy original(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest,
                            p.thickness, p.theta);
    ShellFemEnergy identity(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest,
                            p.thickness, p.theta, std::vector<Eigen::Matrix3d>(4, Eigen::Matrix3d::Identity()));
    EXPECT_NEAR(identity.eval_value(u), original.eval_value(u), 1e-12);
    EXPECT_TRUE(identity.eval_gradient(u).isApprox(original.eval_gradient(u), 1e-12));
    EXPECT_DOUBLE_EQ(elastic.max_feasible_step(u, -2 * X), original.max_feasible_step(u, -2 * X));
    double expected = 0;
    for (int e = 0; e < 4; ++e) {
        const auto& r = p.rest[e];
        const Eigen::MatrixXd Y = r.X_local * P.transpose();
        TriangleMidEdgeRestData natural(Y, r.opposite);
        natural.sqrt_det_a_bar = r.sqrt_det_a_bar;
        expected +=
            evaluate_shell_element<KoiterStVK>(natural, r.X_local + p.map.gather(e, u) - Y, p.thickness[e], p.theta[e]);
    }
    EXPECT_NEAR(elastic.eval_value(u), expected, 1e-12);
    ShellFemForceFpResponse response(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(),
                                     dof::ConstantDofMap(topology(), 9), p.map, p.rest, u, p.thickness, p.theta);
    EXPECT_TRUE(elastic.eval_gradient(u).isApprox(response.evaluate(P.reshaped())->value(), 1e-11));
    energy::test::expect_derivatives_match(elastic, u);
    plastic[0].setZero();
    p.rest.clear();
    EXPECT_NEAR(elastic.eval_value(u), expected, 1e-12);
}

TEST(ShellFemEnergy, RejectsInvalidPlasticFields) {
    Patch p;
    using Shell = ShellFemEnergy;
    auto malformed = p.rest;
    malformed[0].X_local.resize(malformed[0].X_local.rows(), 2);
    EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, malformed,
                        p.thickness, p.theta, std::vector<Eigen::Matrix3d>(4, Eigen::Matrix3d::Identity()))),
                 std::invalid_argument);

    EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest, p.thickness,
                        p.theta, {Eigen::Matrix3d::Identity()})),
                 std::invalid_argument);
    for (const double diagonal :
         {0., -1., std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        std::vector<Eigen::Matrix3d> plastic(4, Eigen::Matrix3d::Identity());
        plastic[2](0, 0) = diagonal;
        EXPECT_THROW((Shell(std::make_shared<litepgo::fem::PointwiseShellAdapter<KoiterStVK>>(), p.map, p.rest,
                            p.thickness, p.theta, plastic)),
                     std::domain_error);
    }
}

} // namespace

} // namespace litepgo::fem::test
