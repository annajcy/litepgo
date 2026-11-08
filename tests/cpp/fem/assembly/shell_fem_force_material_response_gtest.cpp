#include "litepgo/fem/assembly/detail/shell_fp_force.hpp"
#include "litepgo/fem/assembly/shell_fem_energy.hpp"
#include "litepgo/fem/assembly/shell_fem_force_material_response.hpp"
#include "litepgo/fem/material/shell/koiter_stvk.hpp"
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

TEST(ShellForceEvaluation, OverwritesStridedViewsAndAppliesOriginalAreaOnce) {
    Patch p;
    const auto u = p.map.gather(0, p.u);
    const Vec theta = Eigen::Vector2d(2, 3);
    const detail::ShellForceData material(p.rest[0], u, p.thickness[0]);
    const detail::ShellFpForceData plastic(p.rest[0], u, p.thickness[0], theta);
    Vec expected(u.size());
    StridedVecRef expected_view(expected);
    detail::eval_shell_element<KoiterStVK>(p.rest[0], u, p.thickness[0], theta, &expected_view);
    const Vec identity = Eigen::Matrix3d::Identity().eval().reshaped();
    for (bool fp : {false, true}) {
        const int n = static_cast<int>(u.size()), width = fp ? 9 : 2;
        Vec storage = Vec::Constant(2 * n + 2, -123);
        Eigen::Map<Vec, 0, Eigen::InnerStride<2>> mapped(storage.data() + 1, n);
        StridedVecRef value(mapped);
        Eigen::MatrixXd buffer = Eigen::MatrixXd::Constant(n + 2, width + 2, -123);
        StridedMatRef J(buffer.block(1, 1, n, width));
        auto eval = [&](StridedMatRef* derivative) {
            if (fp)
                detail::eval_shell_fp_force<KoiterStVK>(plastic, identity, value, derivative);
            else
                detail::eval_shell_material_force<KoiterStVK>(material, theta, value, derivative);
        };
        eval(&J);
        EXPECT_TRUE(value.isApprox(expected, 1e-12));
        const Eigen::MatrixXd first = J;
        value.setConstant(std::numeric_limits<double>::quiet_NaN());
        J.setConstant(std::numeric_limits<double>::quiet_NaN());
        eval(&J);
        EXPECT_TRUE(value.isApprox(expected, 1e-12));
        EXPECT_EQ(J, first);
        value.setConstant(99);
        eval(nullptr);
        EXPECT_TRUE(value.isApprox(expected, 1e-12));
        EXPECT_EQ(J, first);
        for (int i = 0; i < storage.size(); ++i)
            if (i % 2 == 0 || i == 2 * n + 1)
                EXPECT_EQ(storage[i], -123);
        EXPECT_TRUE(buffer.row(0).isConstant(-123));
        EXPECT_TRUE(buffer.row(n + 1).isConstant(-123));
        EXPECT_TRUE(buffer.col(0).isConstant(-123));
        EXPECT_TRUE(buffer.col(width + 1).isConstant(-123));
    }
}
TEST(ShellForceEvaluation, RejectsInvalidOriginalAreaForBothCaches) {
    Patch p;
    for (double bad : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        p.rest[0].sqrt_det_a_bar = bad;
        EXPECT_THROW((detail::ShellForceData(p.rest[0], p.map.gather(0, p.u), .1)), std::invalid_argument);
        EXPECT_THROW((detail::ShellFpForceData(p.rest[0], p.map.gather(0, p.u), .1, Eigen::Vector2d(2, 3))),
                     std::invalid_argument);
    }
}
struct NonFiniteStress {
    static constexpr int num_parameters = KoiterStVK::num_parameters;
    static constexpr auto eval_d2W_dab_dtheta = KoiterStVK::eval_d2W_dab_dtheta;
    static constexpr auto eval_d2W_dab_dabarbbar = KoiterStVK::eval_d2W_dab_dabarbbar;
    static Vec eval_dW_dab(const ShellMaterialInputs&) {
        return Vec::Constant(6, std::numeric_limits<double>::infinity());
    }
};
struct NonFiniteMixed {
    static constexpr int num_parameters = KoiterStVK::num_parameters;
    static constexpr auto eval_dW_dab = KoiterStVK::eval_dW_dab;
    static Eigen::MatrixXd eval_d2W_dab_dtheta(const ShellMaterialInputs&) {
        return Eigen::MatrixXd::Constant(6, 2, std::numeric_limits<double>::quiet_NaN());
    }
    static Eigen::MatrixXd eval_d2W_dab_dabarbbar(const ShellMaterialInputs&) {
        return Eigen::MatrixXd::Constant(6, 6, std::numeric_limits<double>::quiet_NaN());
    }
};
TEST(ShellForceEvaluation, PreservesMaterialDerivativeChecks) {
    Patch p;
    const auto u = p.map.gather(0, p.u);
    const Vec theta = Eigen::Vector2d(2, 3), identity = Eigen::Matrix3d::Identity().eval().reshaped();
    const detail::ShellForceData material(p.rest[0], u, .1);
    const detail::ShellFpForceData plastic(p.rest[0], u, .1, theta);
    Vec output(u.size());
    StridedVecRef value(output);
    Eigen::MatrixXd mixed(u.size(), 2), plastic_mixed(u.size(), 9);
    StridedMatRef J(mixed), P(plastic_mixed);
    EXPECT_THROW(detail::eval_shell_material_force<NonFiniteStress>(material, theta, value, nullptr),
                 std::domain_error);
    EXPECT_THROW(detail::eval_shell_fp_force<NonFiniteStress>(plastic, identity, value, nullptr), std::domain_error);
    EXPECT_THROW(detail::eval_shell_material_force<NonFiniteMixed>(material, theta, value, &J), std::domain_error);
    EXPECT_THROW(detail::eval_shell_fp_force<NonFiniteMixed>(plastic, identity, value, &P), std::domain_error);
}
} // namespace
} // namespace litepgo::fem::test
