#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/kinematics/positive_det_feasible_step.hpp"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace litepgo::fem::test {

namespace {

void check_segment(const Eigen::Matrix3d& A, const Eigen::Matrix3d& dA, double alpha) {
    ASSERT_GE(alpha, 0);
    ASSERT_LE(alpha, 1);
    for (int i = 0; i <= 100; ++i) {
        const Eigen::Matrix3d trial = A + (alpha * i / 100) * dA;
        EXPECT_GT(trial.determinant(), 0);
        EXPECT_TRUE(std::isfinite(trial.determinant()));
    }
}

TEST(PositiveDetFeasibleStep, ZeroExpansionShearAndRotationDirectionsAcceptFullStep) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d shear = Eigen::Matrix3d::Zero();
    shear(0, 1) = 10;
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Zero();
    rotation(0, 1) = -4;
    rotation(1, 0) = 4;
    for (const Eigen::Matrix3d& dA : {Eigen::Matrix3d::Zero().eval(), I, shear, rotation}) {
        EXPECT_EQ(eval_positive_det_feasible_step(I, dA), 1);
        check_segment(I, dA, 1);
    }
}

TEST(PositiveDetFeasibleStep, HandlesLinearQuadraticAndCubicPolynomials) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    for (const Eigen::Vector3d& diagonal :
         {Eigen::Vector3d(-4, 0, 0), Eigen::Vector3d(-4, -2, 0), Eigen::Vector3d(-4, -2, -1)}) {
        const Eigen::Matrix3d dA = diagonal.asDiagonal();
        const double alpha = eval_positive_det_feasible_step(I, dA);
        EXPECT_NEAR(alpha, 0.225, 1e-12);
        check_segment(I, dA, alpha);
    }
    EXPECT_EQ(eval_positive_det_feasible_step(I, -0.5 * I), 1);
    Eigen::Matrix3d cyclic;
    cyclic << 0, 1, 0, 0, 0, 1, -8, 0, 0;
    const double alpha = eval_positive_det_feasible_step(I, cyclic); // det(I+t*cyclic)=1-8*t^3.
    EXPECT_NEAR(alpha, 0.45, 1e-12);
    check_segment(I, cyclic, alpha);
}

TEST(PositiveDetFeasibleStep, FindsInteriorDoubleRootWithPositiveEndpointAndTripleRoots) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    for (const Eigen::Vector3d& diagonal :
         {Eigen::Vector3d(-4, -4, 0), Eigen::Vector3d(-4, -4, 1), Eigen::Vector3d(-4, -4, -4)}) {
        const Eigen::Matrix3d dA = diagonal.asDiagonal();
        EXPECT_EQ((I + 0.25 * dA).determinant(), 0);
        const double alpha = eval_positive_det_feasible_step(I, dA);
        EXPECT_NEAR(alpha, 0.225, 2e-5);
        EXPECT_LT(alpha, 0.25);
        check_segment(I, dA, alpha);
    }
}

TEST(PositiveDetFeasibleStep, IsInvariantUnderScalingAndChangesOfBasis) {
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    Eigen::Matrix3d B;
    B << 1.2, 0.2, -0.1, 0.1, 0.9, 0.2, 0.02, -0.1, 1.1;
    const Eigen::Matrix3d D = Eigen::Vector3d(-4, -2, 1).asDiagonal();
    for (double scale : {1e-50, 0.1, 1.0, 10.0, 1e50}) {
        const Eigen::Matrix3d A = scale * R * B, dA = scale * R * D * B;
        const double alpha = eval_positive_det_feasible_step(A, dA);
        EXPECT_NEAR(alpha, 0.225, 1e-12);
        check_segment(A, dA, alpha);
    }
    Eigen::Matrix3d extreme = Eigen::Matrix3d::Zero();
    extreme(0, 0) = -1e200;
    const double tiny = eval_positive_det_feasible_step(Eigen::Matrix3d::Identity(), extreme);
    EXPECT_NEAR(tiny / 9e-201, 1, 1e-12);
    check_segment(Eigen::Matrix3d::Identity(), extreme, tiny);
    EXPECT_EQ(eval_positive_det_feasible_step(Eigen::Matrix3d::Identity(), 1e300 * Eigen::Matrix3d::Identity()), 0);
}

TEST(PositiveDetFeasibleStep, RejectsInvalidStartsAndNonFiniteDirections) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity(), Z = Eigen::Matrix3d::Zero();
    EXPECT_THROW(eval_positive_det_feasible_step(Z, I), std::domain_error);
    EXPECT_THROW(eval_positive_det_feasible_step(-I, Z), std::domain_error);
    EXPECT_THROW(eval_positive_det_feasible_step(1e200 * I, Z), std::domain_error);
    for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        Eigen::Matrix3d bad = I;
        bad(0, 0) = value;
        EXPECT_THROW(eval_positive_det_feasible_step(bad, Z), std::invalid_argument);
        EXPECT_THROW(eval_positive_det_feasible_step(I, bad), std::invalid_argument);
    }
}
} // namespace

} // namespace litepgo::fem::test
