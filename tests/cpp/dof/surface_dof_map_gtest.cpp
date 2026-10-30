#include "litepgo/dof/surface_dof_map.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

namespace litepgo::dof::test {

namespace {

// Three surface vertices with distinct rest positions.
Vec three_rest_positions() {
    return (Vec(9) << 0, 0, 0, 1, 0, 0, 0, 1, 0).finished();
}

} // namespace

TEST(SurfaceDofMap, SelectsVertexDofs) {
    // Surface vertices take simulation vertices 4 and 1; the last one is static.
    const SurfaceDofMap map(three_rest_positions(), {4, 1, -1}, 18);
    EXPECT_EQ(map.num_vertices(), 3);
    EXPECT_EQ(map.num_dofs(), 18);
    EXPECT_EQ(map.rest_positions(), three_rest_positions());
    EXPECT_EQ(map.displacement_map().nonZeros(), 6);

    const Vec u = Vec::LinSpaced(18, 1, 18);
    Vec expected = Vec::Zero(9);
    expected.segment<3>(0) = u.segment<3>(12);
    expected.segment<3>(3) = u.segment<3>(3);
    EXPECT_EQ(map.displacements(u), expected);
    EXPECT_EQ(map.positions(u), three_rest_positions() + expected);
}

TEST(SurfaceDofMap, SelectsTheValueSlotOfWiderVertices) {
    // Hex Hermite: 24 DOFs per vertex, positions in the first three.
    const SurfaceDofMap map(three_rest_positions(), {2, 0, 1}, 3 * 24, 24);
    const Vec u = Vec::LinSpaced(72, 0, 71);
    EXPECT_EQ(map.displacements(u), (Vec(9) << 48, 49, 50, 0, 1, 2, 24, 25, 26).finished());
}

TEST(SurfaceDofMap, KeepsAGeneralMap) {
    // The middle surface vertex is embedded halfway between simulation vertices 0 and 1.
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(9, 6);
    W.block<3, 3>(0, 0).setIdentity();
    W.block<3, 3>(3, 0) = 0.5 * Eigen::Matrix3d::Identity();
    W.block<3, 3>(3, 3) = 0.5 * Eigen::Matrix3d::Identity();
    W.block<3, 3>(6, 3).setIdentity();
    const SurfaceDofMap map(three_rest_positions(), SpMat(W.sparseView()));
    const Vec u = Vec::LinSpaced(6, 1, 6);
    EXPECT_EQ(map.displacements(u), W * u);
    EXPECT_EQ(Eigen::MatrixXd(map.displacement_map()), W);
}

TEST(SurfaceDofMap, RejectsInvalidInput) {
    EXPECT_THROW(SurfaceDofMap(Vec::Zero(8), SpMat(8, 3)), std::invalid_argument);
    EXPECT_THROW(SurfaceDofMap(three_rest_positions(), SpMat(6, 3)), std::invalid_argument);
    Vec infinite = three_rest_positions();
    infinite[4] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(SurfaceDofMap(infinite, SpMat(9, 3)), std::invalid_argument);

    EXPECT_THROW(SurfaceDofMap(three_rest_positions(), {0, 1}, 9), std::invalid_argument);       // too few vertices
    EXPECT_THROW(SurfaceDofMap(three_rest_positions(), {0, 1, 3}, 9), std::invalid_argument);    // beyond num_dofs
    EXPECT_THROW(SurfaceDofMap(three_rest_positions(), {0, 1, -2}, 9), std::invalid_argument);   // below -1
    EXPECT_THROW(SurfaceDofMap(three_rest_positions(), {0, 1, 2}, 9, 2), std::invalid_argument); // too narrow
    EXPECT_THROW(SurfaceDofMap(three_rest_positions(), {-1, -1, 0}, 0), std::invalid_argument);  // no DOFs at all

    const SurfaceDofMap map(three_rest_positions(), {0, 1, 2}, 9);
    EXPECT_THROW(map.displacements(Vec::Zero(8)), std::invalid_argument);
}

} // namespace litepgo::dof::test
