#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/fem/basis/2d/triangle_p1_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"
#include <memory>

#include <gtest/gtest.h>

#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litepgo::dof::test {

namespace {
static_assert(std::same_as<decltype(std::declval<const StencilDofMap&>().gather(0, std::declval<const Vec&>())),
                           Eigen::MatrixXd>);

TEST(StencilDofMap, GathersVariableStencilsInVertexAndSlotOrder) {
    const std::vector<std::vector<int>> stencils{{4, 1, 3, 0}, {2}, {}, {1, 4, 0}};
    for (int width : {1, 3}) {
        const StencilDofMap map(6, stencils, width);
        const Vec data = Vec::LinSpaced(map.num_dofs() + 2, 0, map.num_dofs() + 1);
        EXPECT_EQ(map.num_elements(), 4);
        EXPECT_EQ(map.num_dofs(), 6 * width);
        EXPECT_EQ(map.dofs_per_vertex(), width);
        for (int e = 0; e < map.num_elements(); ++e) {
            SCOPED_TRACE(e);
            EXPECT_EQ(map.vertex_indices(e), stencils[static_cast<std::size_t>(e)]);
            const auto local = map.gather(e, data.segment(1, map.num_dofs()));
            const auto indices = map.dof_indices(e);
            EXPECT_EQ(local.rows(), stencils[static_cast<std::size_t>(e)].size());
            EXPECT_EQ(local.cols(), width);
            const Vec flat = local.reshaped<Eigen::RowMajor>();
            ASSERT_EQ(indices.size(), flat.size());
            for (Eigen::Index j = 0; j < flat.size(); ++j)
                EXPECT_DOUBLE_EQ(flat[j], data[indices[static_cast<std::size_t>(j)] + 1]);
        }
    }
    const StencilDofMap map(6, stencils, 3);
    EXPECT_EQ(map.dof_indices(0), (std::vector<int>{12, 13, 14, 3, 4, 5, 9, 10, 11, 0, 1, 2}));
    Eigen::Matrix<double, 4, 3> expected;
    expected << 12, 13, 14, 3, 4, 5, 9, 10, 11, 0, 1, 2;
    EXPECT_EQ(map.gather(0, Vec::LinSpaced(18, 0, 17)), expected);
}

TEST(StencilDofMap, OwnsConnectivityAndReturnsIndependentCurrentValues) {
    std::vector<std::vector<int>> stencils{{3, 0, 2}};
    const StencilDofMap map(4, stencils, 2);
    stencils[0] = {1};
    Vec data = Vec::LinSpaced(8, 0, 7);
    auto local = map.gather(0, data);
    const auto snapshot = local;
    data[6] += 10;
    EXPECT_DOUBLE_EQ(local(0, 0), 6);
    EXPECT_DOUBLE_EQ(map.gather(0, data)(0, 0), 16);
    local.setZero();
    EXPECT_DOUBLE_EQ(data[6], 16);
    const auto temporary_map = [] { return StencilDofMap(4, {{3, 0, 2}}, 2); }();
    EXPECT_EQ(temporary_map.vertex_indices(0), (std::vector<int>{3, 0, 2}));
    EXPECT_EQ(temporary_map.gather(0, Vec::LinSpaced(8, 0, 7)), snapshot);
}

TEST(StencilDofMap, ValidatesLayoutAndStencilVertices) {
    EXPECT_THROW((StencilDofMap(-1, {}, 3)), std::invalid_argument);
    for (int width : {0, -1, std::numeric_limits<int>::max()})
        EXPECT_THROW((StencilDofMap(3, {{0, 1, 2}}, width)), std::invalid_argument);
    EXPECT_THROW((StencilDofMap(std::numeric_limits<int>::max(), {}, 2)), std::invalid_argument);
    for (const std::vector<int>& vertices : {std::vector<int>{-1}, {3}, {0, 1, 0}})
        EXPECT_THROW((StencilDofMap(3, {vertices}, 3)), std::invalid_argument);
    EXPECT_THROW((StencilDofMap(0, {{0}}, 3)), std::invalid_argument);
}

TEST(StencilDofMap, ChecksElementAndDataSizesIncludingEmptyStencils) {
    const StencilDofMap map(4, {{2, 0}, {}}, 3);
    const Vec data = Vec::Zero(12);
    for (int e : {-1, 2}) {
        EXPECT_THROW(map.vertex_indices(e), std::out_of_range);
        EXPECT_THROW(map.dof_indices(e), std::out_of_range);
        EXPECT_THROW(map.gather(e, data), std::out_of_range);
    }
    for (int size : {0, 11, 13}) {
        EXPECT_THROW(map.gather(0, Vec::Zero(size)), std::invalid_argument);
        EXPECT_THROW(map.gather(1, Vec::Zero(size)), std::invalid_argument);
    }
    EXPECT_EQ(map.gather(1, data).rows(), 0);
    EXPECT_EQ(map.gather(1, data).cols(), 3);
    EXPECT_TRUE(map.dof_indices(1).empty());
    const StencilDofMap empty(0, {}, 3);
    EXPECT_EQ(empty.num_elements(), 0);
    EXPECT_EQ(empty.num_dofs(), 0);
    EXPECT_THROW(empty.gather(0, Vec::Zero(0)), std::out_of_range);
    const StencilDofMap no_active_vertices(0, {{}}, 3);
    EXPECT_EQ(no_active_vertices.gather(0, Vec::Zero(0)).rows(), 0);
    EXPECT_EQ(StencilDofMap(2, {}, 3).num_dofs(), 6);
}

TEST(StencilDofMap, MidEdgeStencilsKeepCentralFaceInterpolationSeparateFromNeighborSupport) {
    const mesh::TriangleTopology topology(6, {{4, 1, 5}, {5, 1, 3}, {4, 5, 0}, {1, 4, 2}});
    const auto stencils = mesh::make_triangle_mid_edge_stencils(topology);
    const StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    const Vec rest = Vec::LinSpaced(map.num_dofs(), 0, map.num_dofs() - 1);
    Vec displacement = 0.1 * rest;
    const auto X_local = map.gather(0, rest);
    const auto u_local = map.gather(0, displacement);
    ASSERT_EQ(u_local.rows(), 6);
    auto X_field = std::make_shared<fem::InterpolatedResponse<fem::TriangleP1Basis>>(X_local.topRows(3));
    auto u_field = std::make_shared<fem::InterpolatedResponse<fem::TriangleP1Basis>>(u_local.topRows(3));
    const Eigen::Vector2d xi(0.2, 0.3);
    const Eigen::Vector3d expected =
        0.5 * u_local.row(0).transpose() + 0.2 * u_local.row(1).transpose() + 0.3 * u_local.row(2).transpose();
    EXPECT_TRUE(u_field->eval_value(xi).isApprox(expected, 1e-14));
    const int neighbor_row = stencils.opposite[0][0];
    const int neighbor = stencils.vertices[0][static_cast<std::size_t>(neighbor_row)];
    displacement[3 * neighbor] += 1;
    const auto changed = map.gather(0, displacement);
    EXPECT_DOUBLE_EQ(changed(neighbor_row, 0), u_local(neighbor_row, 0) + 1);
    EXPECT_EQ(changed.topRows(3), u_local.topRows(3));
    EXPECT_EQ(map.gather(0, rest), X_local);
    EXPECT_TRUE(X_field->eval_value(xi).isApprox(10 * expected, 1e-14));
}
} // namespace

} // namespace litepgo::dof::test
