#include "litepgo/fem/dof/constant_dof_map.hpp"
#include <gtest/gtest.h>

namespace {
using namespace litepgo;

TEST(ConstantDofMap, EveryElementUsesTheSameIndependentSnapshotAndIndices) {
    const HexTopology topology(8, {{7, 6, 5, 4, 3, 2, 1, 0}, {0, 1, 2, 3, 4, 5, 6, 7}});
    const ConstantDofMap map(topology);
    ConstantDofData<2> data(Eigen::Vector2d(2, 3));
    for (int e = 0; e < 2; ++e) {
        auto local = map.gather(e, data);
        EXPECT_EQ(local, Eigen::Vector2d(2, 3));
        EXPECT_EQ(map.dof_indices(e, data), (std::array<int, 2>{0, 1}));
        local.setZero();
        EXPECT_EQ(data.values(), Eigen::Vector2d(2, 3));
    }
    const auto snapshot = map.gather(0, data);
    data.values() << 4, 5;
    EXPECT_EQ(snapshot, Eigen::Vector2d(2, 3));
    EXPECT_EQ(map.gather(1, data), Eigen::Vector2d(4, 5));
    Eigen::Matrix<double, 2, 2, Eigen::RowMajor> matrix;
    matrix << 1, 2, 3, 4;
    const ConstantDofData<4> matrix_data(matrix.reshaped().eval());
    const auto local = map.gather(1, matrix_data);
    const auto indices = map.dof_indices(1, matrix_data);
    for (int j = 0; j < 4; ++j) {
        EXPECT_EQ(indices[static_cast<std::size_t>(j)], j);
        EXPECT_EQ(local[j], matrix_data.values()[indices[static_cast<std::size_t>(j)]]);
    }
}

TEST(ConstantDofMap, ChecksIndicesIncludingEmptyTopology) {
    const ConstantDofMap map(TetTopology(4, {{0, 1, 2, 3}}));
    const ConstantDofData<2> data(Eigen::Vector2d(2, 3));
    for (int e : {-1, 1}) {
        EXPECT_THROW(map.gather(e, data), std::out_of_range);
        EXPECT_THROW(map.dof_indices(e, data), std::out_of_range);
    }
    const ConstantDofMap empty(TriangleTopology(0, {}));
    EXPECT_THROW(empty.gather(0, data), std::out_of_range);
    EXPECT_THROW(empty.dof_indices(0, data), std::out_of_range);
}

TEST(ConstantDofMap, OwnsCountAfterTopologyDestruction) {
    const auto map = [] {
        const TriangleTopology topology(3, {{0, 1, 2}});
        return ConstantDofMap(topology);
    }();
    const ConstantDofData<9> data(Eigen::Matrix3d::Identity().eval().reshaped().eval());
    EXPECT_TRUE(map.gather(0, data).reshaped(3, 3).isIdentity());
    EXPECT_EQ(map.dof_indices(0, data), (std::array<int, 9>{0, 1, 2, 3, 4, 5, 6, 7, 8}));
}
} // namespace
