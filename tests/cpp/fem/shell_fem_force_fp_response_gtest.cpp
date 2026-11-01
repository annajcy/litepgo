#include "litepgo/fem/material/shell/koiter_stvk.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include "litepgo/fem/shell_fem_force_fp_response.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include <gtest/gtest.h>

namespace litepgo::fem::test {
namespace {
using Force = ShellFemForceFpResponse<KoiterStVK>;
struct Patch {
    mesh::TriangleTopology topology{6, {{0, 1, 2}, {2, 1, 3}, {0, 2, 4}, {1, 0, 5}}};
    dof::StencilDofMap map{6, mesh::make_triangle_mid_edge_stencils(topology).vertices, 3};
    std::vector<TriangleMidEdgeRestData> rest;
    std::vector<double> thickness{.1, .2, .15, .3};
    std::vector<Vec> theta{Eigen::Vector2d(2, 3), Eigen::Vector2d(4, 5), Eigen::Vector2d(-.5, 2),
                           Eigen::Vector2d(8, 9)};
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
    Vec force(const Vec& plastic) const {
        Vec result = Vec::Zero(18);
        for (int e = 0; e < 4; ++e) {
            const Eigen::Matrix3d P = plastic.segment(plastic.size() == 9 ? 0 : 9 * e, 9).reshaped(3, 3);
            const Eigen::MatrixXd Y = rest[e].X_local * P.transpose();
            const TriangleMidEdgeRestData natural(Y, rest[e].opposite);
            const Eigen::MatrixXd natural_u = rest[e].X_local + map.gather(e, u) - Y;
            const ShellMaterialLocalEnergy2D<KoiterStVK> local(natural, natural_u, thickness[e], theta[e]);
            const Vec f = (.5 * rest[e].sqrt_det_a_bar) * local.eval_gradient(Eigen::Vector2d::Constant(1. / 3));
            const auto ids = map.dof_indices(e);
            for (int j = 0; j < f.size(); ++j)
                result[ids[j]] += f[j];
        }
        return result;
    }
};
TEST(ShellFemForceFpResponse, OriginalAreaAssemblyAndFullJacobianForSharedAndPerFaceFp) {
    Patch p;
    Eigen::Matrix3d P;
    P << 1.12, .07, -.02, .03, .94, .06, .01, -.03, 1.08;
    const Vec shared_P = P.reshaped();
    Vec per(36);
    for (int e = 0; e < 4; ++e)
        per.segment<9>(9 * e) = ((1 + .02 * e) * P).eval().reshaped();
    Eigen::MatrixXd previous;
    for (int batch : {1, 3, 256}) {
        Force shared(dof::ConstantDofMap(p.topology, 9), p.map, p.rest, p.u, p.thickness, p.theta, batch);
        Force element(dof::ElementDofMap(p.topology, 9), p.map, p.rest, p.u, p.thickness, p.theta, batch);
        EXPECT_TRUE(shared.eval_value(shared_P).isApprox(p.force(shared_P), 1e-11));
        EXPECT_TRUE(element.eval_value(per).isApprox(p.force(per), 1e-11));
        const Eigen::MatrixXd J = shared.eval_jacobian(shared_P), E = element.eval_jacobian(per);
        if (previous.size())
            EXPECT_EQ((J - previous).norm(), 0);
        previous = J;
        for (int j = 0; j < 36; ++j) {
            Vec plus = per, minus = per;
            plus[j] += 1e-6;
            minus[j] -= 1e-6;
            EXPECT_LT((E.col(j) - (element.eval_value(plus) - element.eval_value(minus)) / 2e-6).norm(), 2e-7);
        }
        for (int j = 0; j < 9; ++j) {
            Vec plus = shared_P, minus = shared_P;
            plus[j] += 1e-6;
            minus[j] -= 1e-6;
            EXPECT_LT((J.col(j) - (p.force(plus) - p.force(minus)) / 2e-6).norm(), 2e-7);
        }
        Vec repeated(36);
        for (int e = 0; e < 4; ++e)
            repeated.segment<9>(9 * e) = shared_P;
        const Eigen::MatrixXd blocks = element.eval_jacobian(repeated);
        Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(18, 9);
        for (int e = 0; e < 4; ++e)
            sum += blocks.middleCols(9 * e, 9);
        EXPECT_TRUE(sum.isApprox(J, 1e-12));
        EXPECT_THROW(shared.eval_value(Vec::Zero(9)), std::domain_error);
        EXPECT_TRUE(shared.eval_value(shared_P).isApprox(p.force(shared_P), 1e-11));
        EXPECT_FALSE(shared.update_pattern(shared_P));
    }
}
TEST(ShellFemForceFpResponse, FixedInputSnapshotsValidationAndEmptyMesh) {
    Patch p;
    const dof::ConstantDofMap params(p.topology, 9);
    Force response(params, p.map, p.rest, p.u, p.thickness, p.theta);
    const Vec identity = Eigen::Matrix3d::Identity().reshaped();
    const auto expected = response.eval_value(identity);
    EXPECT_THROW((Force(params, p.map, {}, p.u, p.thickness, p.theta)), std::invalid_argument);
    EXPECT_THROW((Force(params, p.map, p.rest, p.u, {}, p.theta)), std::invalid_argument);
    EXPECT_THROW((Force(params, p.map, p.rest, p.u, p.thickness, {})), std::invalid_argument);
    EXPECT_THROW((Force(dof::ConstantDofMap(p.topology, 8), p.map, p.rest, p.u, p.thickness, p.theta)),
                 std::invalid_argument);
    EXPECT_THROW((Force(params, p.map, p.rest, p.u, p.thickness, p.theta, 0)), std::invalid_argument);
    p.theta[0] = Vec::Ones(3);
    EXPECT_THROW((Force(params, p.map, p.rest, p.u, p.thickness, p.theta)), std::invalid_argument);
    p.rest.clear();
    p.u.setZero();
    p.thickness[0] = 100;
    EXPECT_EQ((response.eval_value(identity) - expected).norm(), 0);
    const mesh::TriangleTopology empty(0, {});
    Force zero(dof::ConstantDofMap(empty, 9), dof::StencilDofMap(0, {}, 3), {}, Vec(0), {}, {});
    EXPECT_EQ(zero.eval_value(identity).size(), 0);
    EXPECT_EQ(zero.eval_jacobian(identity).cols(), 9);
}
} // namespace
} // namespace litepgo::fem::test
