#include "litepgo/dof/surface_kinematics.hpp"
#include "litepgo/mesh/extract_surface.hpp"

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

TEST(SurfaceKinematics, KeepsAGeneralMap) {
    // The middle surface vertex is embedded halfway between simulation vertices 0 and 1.
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(9, 6);
    W.block<3, 3>(0, 0).setIdentity();
    W.block<3, 3>(3, 0) = 0.5 * Eigen::Matrix3d::Identity();
    W.block<3, 3>(3, 3) = 0.5 * Eigen::Matrix3d::Identity();
    W.block<3, 3>(6, 3).setIdentity();
    const SurfaceKinematics map(three_rest_positions(), SpMat(W.sparseView()));
    const Vec u = Vec::LinSpaced(6, 1, 6);
    EXPECT_EQ(map.displacements(u), W * u);
    EXPECT_EQ(Eigen::MatrixXd(map.displacement_map()), W);
}

TEST(SurfaceKinematics, RejectsInvalidInput) {
    EXPECT_THROW(SurfaceKinematics(Vec::Zero(8), SpMat(8, 3)), std::invalid_argument);
    EXPECT_THROW(SurfaceKinematics(three_rest_positions(), SpMat(6, 3)), std::invalid_argument);
    Vec infinite = three_rest_positions();
    infinite[4] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(SurfaceKinematics(infinite, SpMat(9, 3)), std::invalid_argument);

    const SurfaceKinematics map(three_rest_positions(), SpMat(9, 9));
    EXPECT_THROW(map.displacements(Vec::Zero(8)), std::invalid_argument);
}

} // namespace litepgo::dof::test
