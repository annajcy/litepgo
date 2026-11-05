#include "litepgo/contact/broad_phase/spatial_hash.hpp"
#include "litepgo/contact/ipc_contact_energy.hpp"
#ifdef LITEPGO_HAS_IPC_TOOLKIT
#include "litepgo/contact/ipctk_contact_energy.hpp"
#endif

#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <memory>
#include <type_traits>

namespace litepgo::contact::test {
namespace {
mesh::TriangleMesh surface() {
    Vec x(18);
    x << 0, 0, 0, 10, 0, 0, 0, 10, 0, 2, 2, .5, 6, 2, 4.5, 2, 6, 4.5;
    return {x, mesh::TriangleTopology(6, {{0, 1, 2}, {3, 4, 5}})};
}
ContactPairWeights uniform(const mesh::TriangleMesh& s, double scale) {
    return {std::vector<int>(s.topology.num_vertices(), 0), std::vector<int>(contact_edges(s.topology).size(), 0),
            std::vector<int>(s.topology.num_cells(), 0), Eigen::MatrixXd::Constant(1, 1, scale)};
}
template <class Backend>
auto make(const mesh::TriangleMesh& s, std::optional<ContactPairWeights> weights = std::nullopt, bool area = true) {
    if constexpr (std::is_same_v<Backend, IPCContactEnergy>) {
        IPCContactOptions options;
        options.dhat = .7;
        options.min_distance = .05;
        options.use_area_weighting = area;
        return std::make_unique<Backend>(s, options, std::make_shared<SpatialHash>(), std::move(weights));
    }
#ifdef LITEPGO_HAS_IPC_TOOLKIT
    else {
        IPCTKContactOptions options;
        options.dhat = .7;
        options.min_distance = .05;
        options.use_area_weighting = area;
        return std::make_unique<Backend>(s, options, std::move(weights));
    }
#endif
}
template <class T> class ContactPairWeightsTest : public ::testing::Test {};
#ifdef LITEPGO_HAS_IPC_TOOLKIT
using Backends = ::testing::Types<IPCContactEnergy, IPCTKContactEnergy>;
#else
using Backends = ::testing::Types<IPCContactEnergy>;
#endif
TYPED_TEST_SUITE(ContactPairWeightsTest, Backends);

TEST(ContactPairWeights, StableEdgeNumbering) {
    const std::vector<std::array<int, 2>> expected{{0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}};
    EXPECT_EQ(contact_edges(mesh::TriangleTopology(4, {{2, 1, 0}, {0, 3, 2}})), expected);
    EXPECT_EQ(contact_edges(mesh::TriangleTopology(4, {{2, 3, 0}, {0, 1, 2}})), expected);
    EXPECT_TRUE(contact_edges(mesh::TriangleTopology(0, {})).empty());
}

TYPED_TEST(ContactPairWeightsTest, UniformWeightsScaleDerivativesAndOwnData) {
    const auto s = surface();
    for (bool area : {false, true}) {
        auto base = make<TypeParam>(s, std::nullopt, area);
        base->update_pattern(s.positions);
        for (double scale : {1.0, .5, 2.0}) {
            auto config = uniform(s, scale);
            auto weighted = make<TypeParam>(s, config, area);
            config.weights.setZero();
            config.vertex_groups.assign(6, 999); // Caller mutations cannot change the energy.
            weighted->update_pattern(s.positions);
            EXPECT_NEAR(weighted->eval_value(s.positions), scale * base->eval_value(s.positions), 1e-12);
            EXPECT_TRUE(weighted->eval_gradient(s.positions).isApprox(scale * base->eval_gradient(s.positions), 1e-12));
            EXPECT_TRUE(Eigen::MatrixXd(weighted->eval_hessian(s.positions))
                            .isApprox(scale * Eigen::MatrixXd(base->eval_hessian(s.positions)), 1e-12));
            EXPECT_TRUE(
                Eigen::MatrixXd(weighted->eval_hessian(s.positions, energy::HessianKind::Projected))
                    .isApprox(scale * Eigen::MatrixXd(base->eval_hessian(s.positions, energy::HessianKind::Projected)),
                              1e-10));
            EXPECT_EQ(weighted->contact_snapshot(s.positions).types.size(),
                      base->contact_snapshot(s.positions).types.size());
        }
    }
}

TYPED_TEST(ContactPairWeightsTest, MixedParentGroupsMergeIntoOneVertexVertexRecord) {
    auto s = surface();
    s.positions << 0, 0, 0, 2, 0, 0, 0, 2, 0, -.2, -.3, .4, -1, -2, 1, -2, -1, 1;
    ContactPairWeights config{{0, 3, 3, 1, 3, 3}, std::vector<int>(6, 3), {1, 2}, Eigen::MatrixXd::Zero(4, 4)};
    config.weights(0, 2) = config.weights(2, 0) = .5;
    config.weights(1, 1) = 2;
    for (bool area : {false, true}) {
        const auto energy = make<TypeParam>(s, config, area);
        const auto snapshot = energy->contact_snapshot(s.positions);
        ASSERT_EQ(snapshot.types, std::vector<std::string>{"VV"});
        const auto triangle_area = [&](int a, int b, int c) {
            const Eigen::Vector3d u = s.positions.segment<3>(3 * b) - s.positions.segment<3>(3 * a);
            const Eigen::Vector3d v = s.positions.segment<3>(3 * c) - s.positions.segment<3>(3 * a);
            return .5 * u.cross(v).norm();
        };
        const double expected = area ? (.5 * triangle_area(0, 1, 2) + 2 * triangle_area(3, 4, 5)) / 12 : 2.5;
        EXPECT_NEAR(snapshot.weights[0], expected, 1e-14);
        EXPECT_EQ(snapshot.vertex_ids.row(0), (Eigen::RowVector4i(0, 3, -1, -1)));
        const double d = snapshot.distances[0] * snapshot.distances[0] - .05 * .05, h = .7 * (.7 + .1);
        EXPECT_NEAR(energy->eval_value(s.positions), -expected * (d - h) * (d - h) * std::log(d / h), 1e-13);
    }
}

TYPED_TEST(ContactPairWeightsTest, ZeroExcludesActiveAndCCDButNotGlobalIntersectionCheck) {
    const auto s = surface();
    auto base = make<TypeParam>(s);
    auto zero = make<TypeParam>(s, uniform(s, 0));
    Vec dx = Vec::Zero(18);
    dx[11] = -2;
    const double step = base->max_feasible_step(s.positions, dx);
    EXPECT_GT(step, 0);
    EXPECT_LT(step, 1);
    EXPECT_DOUBLE_EQ(zero->eval_value(s.positions), 0);
    EXPECT_EQ(zero->eval_gradient(s.positions), Vec::Zero(18));
    EXPECT_FALSE(zero->update_pattern(s.positions));
    EXPECT_EQ(zero->contact_snapshot(s.positions).distances.size(), 0);
    EXPECT_DOUBLE_EQ(zero->max_feasible_step(s.positions, dx), 1);
    for (double scale : {.001, .5, 2.0})
        EXPECT_DOUBLE_EQ(make<TypeParam>(s, uniform(s, scale))->max_feasible_step(s.positions, dx), step);
    Vec intersecting = s.positions;
    intersecting[11] = -.5;
    EXPECT_THROW(zero->max_feasible_step(intersecting, Vec::Zero(18)), std::domain_error);
    intersecting.tail<9>() << 2, 2, 0, 3, 2, 0, 2, 3, 0;
    EXPECT_THROW(zero->max_feasible_step(intersecting, Vec::Zero(18)), std::domain_error);
}

TYPED_TEST(ContactPairWeightsTest, RejectsInvalidGroupsAndTables) {
    const auto s = surface();
    auto config = uniform(s, 1);
    config.edge_groups.pop_back();
    EXPECT_THROW(make<TypeParam>(s, config), std::invalid_argument);
    config = uniform(s, 1);
    config.vertex_groups[0] = -1;
    EXPECT_THROW(make<TypeParam>(s, config), std::invalid_argument);
    config.vertex_groups[0] = 1;
    EXPECT_THROW(make<TypeParam>(s, config), std::invalid_argument);
    config = uniform(s, 1);
    config.weights = Eigen::MatrixXd::Ones(2, 2);
    config.weights(0, 1) = .5;
    EXPECT_THROW(make<TypeParam>(s, config), std::invalid_argument);
    for (double bad : {-1., double(INFINITY), double(NAN)}) {
        config = uniform(s, bad);
        EXPECT_THROW(make<TypeParam>(s, config), std::invalid_argument);
    }
}

TYPED_TEST(ContactPairWeightsTest, CCDSkipsOnlyDisabledPairsInAMixedTable) {
    const auto single = surface();
    Vec x(36);
    x.head<18>() = single.positions;
    x.tail<18>() = single.positions;
    for (int v = 6; v < 12; ++v)
        x[3 * v] += 30;
    const mesh::TriangleMesh doubled{x, mesh::TriangleTopology(12, {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {9, 10, 11}})};
    ContactPairWeights config{{0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2}, {}, {0, 1, 0, 2}, Eigen::MatrixXd::Zero(3, 3)};
    for (auto edge : contact_edges(doubled.topology))
        config.edge_groups.push_back(config.vertex_groups[edge[0]]);
    config.weights(0, 2) = config.weights(2, 0) = .5;
    Vec dx = Vec::Zero(36);
    dx[11] = -2; // Disabled pair would collide first.
    dx[29] = -1; // Enabled pair must still bound the step.
    Vec isolated_dx = Vec::Zero(18);
    isolated_dx[11] = -1;
    const double expected = make<TypeParam>(single)->max_feasible_step(single.positions, isolated_dx);
    const auto weighted = make<TypeParam>(doubled, config);
    EXPECT_EQ(weighted->num_active_pairs(x), 1);
    EXPECT_NEAR(weighted->max_feasible_step(x, dx), expected, 1e-6);
    EXPECT_LT(expected, 1);
    EXPECT_LT(make<TypeParam>(doubled)->max_feasible_step(x, dx), expected);
}
} // namespace
} // namespace litepgo::contact::test
