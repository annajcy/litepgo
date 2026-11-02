#include "litepgo/dof/constant_dof_map.hpp"
#include <gtest/gtest.h>

namespace litepgo::dof::test {

namespace {

TEST(ConstantDofMap, EveryElementUsesTheSameIndependentSnapshotAndIndices) {
    const mesh::HexTopology topology(8, {{7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7}});
    const ConstantDofMap map(topology, 2);
    Vec data(Eigen::Vector2d(2, 3));
    for (int e = 0; e < 2; ++e) {
        auto local = map.gather(e, data);
        EXPECT_EQ(local, Eigen::Vector2d(2, 3));
        EXPECT_EQ(map.dof_indices(e), (std::vector<int>{0, 1}));
        local.setZero();
        EXPECT_EQ(data, Eigen::Vector2d(2, 3));
    }
    const auto snapshot = map.gather(0, data);
    data << 4, 5;
    EXPECT_EQ(snapshot, Eigen::Vector2d(2, 3));
    EXPECT_EQ(map.gather(1, data), Eigen::Vector2d(4, 5));
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> matrix;
    matrix << 1, 2, 3, 4;
    const Vec matrix_data(matrix.reshaped().eval());
    const auto local = ConstantDofMap(topology, 4).gather(1, matrix_data);
    const auto indices = ConstantDofMap(topology, 4).dof_indices(1);
    for (int j = 0; j < 4; ++j) {
        EXPECT_EQ(indices[static_cast<std::size_t>(j)], j);
        EXPECT_EQ(local[j], matrix_data[indices[static_cast<std::size_t>(j)]]);
    }
}

TEST(ConstantDofMap, ChecksIndicesIncludingEmptyTopology) {
    const ConstantDofMap map(mesh::TetTopology(4, {{0, 1, 2, 3}}), 2);
    const Vec data(Eigen::Vector2d(2, 3));
    for (int e : {-1, 1}) {
        EXPECT_THROW(map.gather(e, data), std::out_of_range);
        EXPECT_THROW(map.dof_indices(e), std::out_of_range);
    }
    const ConstantDofMap empty(mesh::TriangleTopology(0, {}), 2);
    EXPECT_THROW(empty.gather(0, data), std::out_of_range);
    EXPECT_THROW(empty.dof_indices(0), std::out_of_range);
}

TEST(ConstantDofMap, OwnsCountAfterTopologyDestruction) {
    const auto map = [] {
        const mesh::TriangleTopology topology(3, {{0, 1, 2}});
        return ConstantDofMap(topology, 9);
    }();
    const Vec data(Eigen::Matrix3d::Identity().eval().reshaped().eval());
    EXPECT_TRUE(map.gather(0, data).reshaped(3, 3).isIdentity());
    EXPECT_EQ(map.dof_indices(0), (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
}
} // namespace

TEST(ConstantDofMap, OwnsRuntimeLayoutAndValidatesWidthAndLength) {
    const litepgo::mesh::TriangleTopology topology(3, {{0, 1, 2}, {2, 1, 0}});
    const litepgo::dof::ConstantDofMap map(topology, 4);
    EXPECT_EQ(map.num_dofs(), 4);
    EXPECT_EQ(map.num_elements(), 2);
    const litepgo::Vec values = litepgo::Vec::LinSpaced(6, 0, 5);
    EXPECT_EQ(map.gather(1, values.segment(1, 4)), values.segment(1, 4));
    for (int width : {0, -1})
        EXPECT_THROW((litepgo::dof::ConstantDofMap(topology, width)), std::invalid_argument);
    for (int size : {3, 5})
        EXPECT_THROW(map.gather(0, litepgo::Vec::Zero(size)), std::invalid_argument);
}

} // namespace litepgo::dof::test
