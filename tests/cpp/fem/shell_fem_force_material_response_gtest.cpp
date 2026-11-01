#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/shell_fem_energy.hpp"
#include "litepgo/fem/shell_fem_force_material_response.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace litepgo::fem::test {
namespace {
using Force = ShellFemForceMaterialResponse<KoiterStVK>;
struct Patch {
    mesh::TriangleTopology topology{6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}}};
    dof::StencilDofMap map{6, mesh::make_triangle_mid_edge_stencils(topology).vertices, 3};
    std::vector<TriangleMidEdgeRestData> rest;
    std::vector<double> thickness{.1, .2, .15, .3};
    Vec u = .03 * Vec::LinSpaced(18, -1, 1);
    Patch() {
        Eigen::Matrix<double, 6, 3> X;
        X << 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, .1, -1, 0, -.1, 0, -2, .2;
        const Vec x = X.reshaped<Eigen::RowMajor>();
        const auto stencil = mesh::make_triangle_mid_edge_stencils(topology);
        for (int e = 0; e < 4; ++e)
            rest.emplace_back(map.gather(e, x), stencil.opposite[e]);
        u[11] += .2;
        u[14] -= .15;
        u[17] += .18;
    }
    Vec force(const Vec& theta) const {
        std::vector<Vec> params;
        for (int e = 0; e < 4; ++e)
            params.push_back(theta.size() == 2 ? theta : Vec(theta.segment<2>(2 * e)));
        return ShellFemEnergy<KoiterStVK>(map, rest, thickness, params).eval_gradient(u);
    }
};
TEST(ShellFemForceMaterialResponse, SharedAndPerFaceParametersMatchEnergyAndFiniteDifferences) {
    Patch p;
    const Eigen::Vector2d theta(2, 3);
    Vec per(8);
    per << 2, 3, 4, 5, -.5, 2, 8, 9;
    Eigen::MatrixXd previous_J;
    for (int batch : {1, 3, 256}) {
        Force shared(dof::ConstantDofMap(p.topology, 2), p.map, p.rest, p.u, p.thickness, batch);
        Force element(dof::ElementDofMap(p.topology, 2), p.map, p.rest, p.u, p.thickness, batch);
        EXPECT_TRUE(shared.has_fixed_pattern());
        const Eigen::MatrixXd J = shared.eval_jacobian(theta), E = element.eval_jacobian(per);
        EXPECT_TRUE(shared.eval_value(theta).isApprox(p.force(theta), 1e-12));
        EXPECT_TRUE(element.eval_value(per).isApprox(p.force(per), 1e-12));
        EXPECT_TRUE((J * theta).isApprox(shared.eval_value(theta), 1e-12));
        EXPECT_TRUE((E * per).isApprox(element.eval_value(per), 1e-12));
        Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(18, 2);
        for (int e = 0; e < 4; ++e)
            sum += E.middleCols(2 * e, 2);
        EXPECT_TRUE(sum.isApprox(J, 1e-12));
        if (previous_J.size())
            EXPECT_EQ((J - previous_J).norm(), 0);
        previous_J = J;
        for (int j = 0; j < 8; ++j) {
            Vec plus = per, minus = per;
            plus[j] += 1e-6;
            minus[j] -= 1e-6;
            EXPECT_LT((E.col(j) - (p.force(plus) - p.force(minus)) / 2e-6).norm(), 1e-8);
        }
        EXPECT_THROW(shared.eval_value(Eigen::Vector2d(1, 0)), std::invalid_argument);
        EXPECT_TRUE(shared.eval_value(theta).isApprox(p.force(theta), 1e-12));
        EXPECT_FALSE(shared.update_pattern(theta));
    }
}
TEST(ShellFemForceMaterialResponse, SnapshotsFixedInputsAndRejectsInvalidConstruction) {
    Patch p;
    dof::ConstantDofMap params(p.topology, 2);
    Force response(params, p.map, p.rest, p.u, p.thickness);
    const Eigen::Vector2d theta(2, 3);
    const auto before = response.eval_value(theta);
    EXPECT_THROW((Force(params, p.map, {}, p.u, p.thickness)), std::invalid_argument);
    EXPECT_THROW((Force(params, p.map, p.rest, p.u, {})), std::invalid_argument);
    EXPECT_THROW((Force(params, p.map, p.rest, Vec::Zero(17), p.thickness)), std::invalid_argument);
    EXPECT_THROW((Force(params, p.map, p.rest, p.u, p.thickness, 0)), std::invalid_argument);
    EXPECT_THROW((Force(dof::ConstantDofMap(p.topology, 3), p.map, p.rest, p.u, p.thickness)), std::invalid_argument);
    p.thickness[0] = 0;
    EXPECT_THROW((Force(params, p.map, p.rest, p.u, p.thickness)), std::invalid_argument);
    p.u.setZero();
    p.rest.clear();
    EXPECT_EQ((response.eval_value(theta) - before).norm(), 0);
    const mesh::TriangleTopology empty(0, {});
    const dof::StencilDofMap map(0, {}, 3);
    Force zero(dof::ConstantDofMap(empty, 2), map, {}, Vec(0), {});
    EXPECT_EQ(zero.eval_value(theta).size(), 0);
    EXPECT_EQ(zero.eval_jacobian(theta).rows(), 0);
    EXPECT_EQ(zero.eval_jacobian(theta).cols(), 2);
}
} // namespace
} // namespace litepgo::fem::test
