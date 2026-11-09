#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>

#include <Eigen/LU>

#include <gtest/gtest.h>
#include <limits>

#include <array>
#include <concepts>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace litepgo::dof::test {

namespace {

using litepgo::mesh::HexTopology;

static_assert(
    std::same_as<decltype(std::declval<const VertexDofMap&>().gather(0, std::declval<const Vec&>())), Eigen::MatrixXd>);
static_assert(std::same_as<decltype(std::declval<const VertexDofMap&>().dof_indices(0)), std::vector<int>>);
static_assert(std::is_constructible_v<VertexDofMap, mesh::HexTopology&&, int>);

// Adjacent cells with a shared face, in Q1 corner order and shuffled global numbering.
mesh::HexTopology two_hexes() {
    return mesh::HexTopology(12, {{1, 3, 9, 4, 7, 6, 0, 10}, {3, 8, 5, 9, 6, 2, 11, 0}});
}

Eigen::Matrix3d reference_jacobian() {
    Eigen::Matrix3d dX_dxi;
    dX_dxi << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    return dX_dxi;
}

Eigen::Matrix<double, 12, 3> rest_positions() {
    Eigen::Matrix<double, 12, 3> nodes;
    nodes << 1, 1, 1, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 1, 1, 0, 0, 1, 1, 2, 1, 1;
    return (nodes * reference_jacobian().transpose()).rowwise() + Eigen::RowVector3d(1.2, -0.7, 2.1);
}

} // namespace

TEST(VertexDofMap, GathersThreeIndependentFieldsInCellAndSlotOrder) {
    const auto topology = two_hexes();
    const VertexDofMap map(topology, 3);

    Vec a = Vec::Zero(36), b = Vec::Zero(36);
    Vec theta = Vec::Zero(24);
    for (int i = 0; i < 36; ++i) {
        a[i] = 100.0 + i;
        b[i] = -1000.0 - 2.0 * i;
    }
    for (int i = 0; i < 24; ++i) {
        theta[i] = 2.0 + 0.1 * i;
    }

    const std::array<std::vector<int>, 2> expected3{{
        {3, 4, 5, 9, 10, 11, 27, 28, 29, 12, 13, 14, 21, 22, 23, 18, 19, 20, 0, 1, 2, 30, 31, 32},
        {9, 10, 11, 24, 25, 26, 15, 16, 17, 27, 28, 29, 18, 19, 20, 6, 7, 8, 33, 34, 35, 0, 1, 2},
    }};
    const std::array<std::vector<int>, 2> expected2{{
        {2, 3, 6, 7, 18, 19, 8, 9, 14, 15, 12, 13, 0, 1, 20, 21},
        {6, 7, 16, 17, 10, 11, 18, 19, 12, 13, 4, 5, 22, 23, 0, 1},
    }};
    for (int e = 0; e < 2; ++e) {
        const auto index = static_cast<std::size_t>(e);
        EXPECT_EQ(map.dof_indices(e), expected3[index]);
        EXPECT_EQ(map.dof_indices(e), expected3[index]);
        EXPECT_EQ(VertexDofMap(topology, 2).dof_indices(e), expected2[index]);
        const auto a_e = map.gather(e, a);
        const auto b_e = map.gather(e, b);
        const auto theta_e = VertexDofMap(topology, 2).gather(e, theta);
        for (int i = 0; i < 8; ++i) {
            for (int c = 0; c < 3; ++c) {
                const int global = expected3[index][static_cast<std::size_t>(3 * i + c)];
                EXPECT_DOUBLE_EQ(a_e(i, c), a[global]);
                EXPECT_DOUBLE_EQ(b_e(i, c), b[global]);
            }
            for (int c = 0; c < 2; ++c) {
                const int global = expected2[index][static_cast<std::size_t>(2 * i + c)];
                EXPECT_DOUBLE_EQ(theta_e(i, c), theta[global]);
            }
        }
    }

    const auto snapshot = map.gather(0, a);
    a[3] += 7.0;
    EXPECT_DOUBLE_EQ(snapshot(0, 0), 103.0);
    EXPECT_DOUBLE_EQ(map.gather(0, a)(0, 0), 110.0);
    EXPECT_DOUBLE_EQ(map.gather(0, b)(0, 0), b[3]);
    auto edited = map.gather(0, a);
    edited(0, 0) = -1.0;
    EXPECT_DOUBLE_EQ(a[3], 110.0);
}

TEST(VertexDofMap, ThreeGatheredFieldsReproduceAffineValuesAndPhysicalGradients) {
    const auto topology = two_hexes();
    const auto X_nodes = rest_positions();
    const VertexDofMap map(topology, 3);

    Eigen::Matrix3d A, P;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    P << 0.1, 0.02, -0.03, -0.04, -0.05, 0.07, 0.06, -0.08, 0.03;
    const Eigen::Vector3d a_offset(-0.3, 0.4, 0.7), b_offset(0.6, -0.2, 0.3);
    Eigen::Matrix<double, 2, 3> M;
    M << 0.4, -0.1, 0.2, -0.3, 0.5, 0.1;
    const Eigen::Vector2d theta_offset(2.0, 3.0);
    Vec a = Vec::Zero(36), b = Vec::Zero(36);
    Vec theta = Vec::Zero(24);
    for (int v = 0; v < 12; ++v) {
        const Eigen::Vector3d position = X_nodes.row(v).transpose();
        a.segment<3>(3 * v) = A * position + a_offset;
        b.segment<3>(3 * v) = P * position + b_offset;
        theta.segment<2>(2 * v) = M * position + theta_offset;
    }

    for (int e = 0; e < 2; ++e) {
        Eigen::Matrix<double, 8, 3> X_e;
        const auto& vertices = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i) {
            X_e.row(i) = X_nodes.row(vertices[static_cast<std::size_t>(i)]);
        }
        auto X_field = std::make_shared<litepgo::fem::InterpolatedResponse<litepgo::fem::HexQ1Basis>>(X_e);
        auto u = std::make_shared<litepgo::fem::InterpolatedResponse<litepgo::fem::HexQ1Basis>>(map.gather(e, a));
        auto up = std::make_shared<litepgo::fem::InterpolatedResponse<litepgo::fem::HexQ1Basis>>(map.gather(e, b));
        auto parameters = std::make_shared<litepgo::fem::InterpolatedResponse<litepgo::fem::HexQ1Basis>>(
            VertexDofMap(topology, 2).gather(e, theta));
        for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(0.76, 0.11, 0.64)}) {
            const Eigen::Vector3d position =
                Eigen::Vector3d(1.2, -0.7, 2.1) + reference_jacobian() * (xi + Eigen::Vector3d(e, 0, 0));
            const auto u_sample_value = u->evaluate(xi)->value();
            const auto u_sample_gradient =
                (u->evaluate(xi)->jacobian() * Eigen::MatrixXd(X_field->evaluate(xi)->jacobian()).inverse()).eval();
            const auto up_sample_value = up->evaluate(xi)->value();
            const auto up_sample_gradient =
                (up->evaluate(xi)->jacobian() * Eigen::MatrixXd(X_field->evaluate(xi)->jacobian()).inverse()).eval();
            const auto theta_sample_value = parameters->evaluate(xi)->value();
            const auto theta_sample_gradient =
                (parameters->evaluate(xi)->jacobian() * Eigen::MatrixXd(X_field->evaluate(xi)->jacobian()).inverse())
                    .eval();
            EXPECT_TRUE(u_sample_value.isApprox(A * position + a_offset, 1e-13));
            EXPECT_TRUE(u_sample_gradient.isApprox(A, 1e-13));
            EXPECT_TRUE(up_sample_value.isApprox(P * position + b_offset, 1e-13));
            EXPECT_TRUE(up_sample_gradient.isApprox(P, 1e-13));
            EXPECT_TRUE(theta_sample_value.isApprox(M * position + theta_offset, 1e-13));
            EXPECT_TRUE(theta_sample_gradient.isApprox(M, 1e-13));
        }
    }
}

TEST(VertexDofMap, SupportsTetConnectivityAndEigenVectorViews) {
    const litepgo::mesh::TetTopology topology(5, {{4, 1, 3, 0}});
    const VertexDofMap map(topology, 2);
    const Vec padded = Vec::LinSpaced(12, -1.0, 10.0);
    const Vec data(padded.segment(1, 10));
    const auto local = map.gather(0, data);
    Eigen::Matrix<double, 4, 2> expected;
    expected << 8, 9, 2, 3, 6, 7, 0, 1;
    EXPECT_EQ(local, expected);
}

TEST(VertexDofMap, GathersScalarVertexDataAsAColumn) {
    const litepgo::mesh::TetTopology topology(5, {{4, 1, 3, 0}});
    const VertexDofMap map(topology, 1);
    const Vec data(Vec::LinSpaced(5, 10.0, 14.0));
    const Eigen::Vector4d expected(14, 11, 13, 10);
    EXPECT_EQ(map.gather(0, data), expected);
}

TEST(VertexDofMap, RejectsMismatchedDataOnEveryCall) {
    const auto topology = two_hexes();
    const VertexDofMap map(topology, 3);
    Vec data = Vec::Zero(36);
    EXPECT_TRUE(map.gather(0, data).isZero());
    for (int count : {11, 13}) {
        data = Vec::Zero(3 * count);
        EXPECT_THROW(map.gather(0, data), std::invalid_argument);
        const Vec parameters = Vec::Zero(2 * count);
        EXPECT_THROW(map.gather(0, parameters), std::invalid_argument);
    }
    data = Vec::Zero(36);
    EXPECT_TRUE(map.gather(0, data).isZero());
}

TEST(VertexDofMap, ChecksElementIndices) {
    const auto topology = two_hexes();
    const VertexDofMap map(topology, 3);
    const Vec data = Vec::Zero(36);
    for (int e : {-1, 2}) {
        EXPECT_THROW(map.dof_indices(e), std::out_of_range);
        EXPECT_THROW(map.gather(e, data), std::out_of_range);
    }
}

TEST(VertexDofMap, AllowsEmptyMeshes) {
    const mesh::HexTopology topology(0, {});
    const VertexDofMap map(topology, 3);
    const Vec data = Vec::Zero(0);
    EXPECT_THROW(map.dof_indices(0), std::out_of_range);
    EXPECT_THROW(map.gather(0, data), std::out_of_range);
}

TEST(VertexDofMap, OwnsConnectivitySnapshot) {
    litepgo::mesh::TriangleTopology topology(4, {{2, 0, 3}});
    const VertexDofMap map(topology, 2);
    topology = litepgo::mesh::TriangleTopology(3, {{0, 1, 2}});
    const Vec data(Vec::LinSpaced(8, 0.0, 7.0));
    Eigen::Matrix<double, 3, 2> expected;
    expected << 4, 5, 0, 1, 6, 7;
    EXPECT_EQ(map.gather(0, data), expected);
    EXPECT_EQ(map.dof_indices(0), (std::vector<int>{4, 5, 0, 1, 6, 7}));

    const VertexDofMap temporary_map(litepgo::mesh::TriangleTopology(4, {{2, 0, 3}}), 2);
    EXPECT_EQ(temporary_map.gather(0, data), expected);
    EXPECT_EQ(temporary_map.dof_indices(0), map.dof_indices(0));
}

TEST(VertexDofMap, OwnsRuntimeLayoutAndChecksWidths) {
    const litepgo::mesh::HexTopology topology(8, {{0, 1, 2, 3, 4, 5, 6, 7}});
    const litepgo::dof::VertexDofMap map(topology, 4);
    EXPECT_EQ(map.num_dofs(), 32);
    EXPECT_EQ(map.num_elements(), 1);
    EXPECT_EQ(map.dofs_per_vertex(), 4);
    const litepgo::Vec padded = litepgo::Vec::LinSpaced(34, 0, 33);
    const auto x_local = map.gather(0, padded.segment(1, 32));
    EXPECT_EQ(x_local.rows(), 8);
    EXPECT_EQ(x_local.cols(), 4);
    EXPECT_EQ(x_local(7, 3), 32);
    for (int width : {0, -1})
        EXPECT_THROW((litepgo::dof::VertexDofMap(topology, width)), std::invalid_argument);
    EXPECT_THROW((litepgo::dof::VertexDofMap(topology, std::numeric_limits<int>::max())), std::invalid_argument);
    for (int count : {31, 33})
        EXPECT_THROW(map.gather(0, litepgo::Vec::Zero(count)), std::invalid_argument);
}

} // namespace litepgo::dof::test
