#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/multiplicative_kinematics.hpp"

#include <Eigen/LU>
#include <gtest/gtest.h>
#include <limits>

namespace litepgo {
namespace {

TEST(MultiplicativeKinematics, IdentityAndOwnedSnapshot) {
    Mat3 F = Mat3::Random();
    const Mat3 original = F;
    MultiplicativeKinematics k(F, Mat3::Identity());
    F.setZero();
    EXPECT_TRUE(k.elastic_gradient().isApprox(original));
    EXPECT_DOUBLE_EQ(k.plastic_volume_ratio(), 1.0);
    EXPECT_DOUBLE_EQ(k.reference_energy(2.5), 2.5);
    EXPECT_TRUE(k.reference_gradient_F(original).isApprox(original));
    const Mat9 H = Mat9::Random();
    EXPECT_TRUE(k.reference_hessian_F(H).isApprox(H));
}

TEST(MultiplicativeKinematics, PullbackMatchesFiniteDifferences) {
    Mat3 Fp;
    Fp << 1.4, 0.2, -0.1, 0.1, 0.8, 0.3, 0.0, -0.2, 1.2;
    Mat3 F;
    F << 1.1, 0.3, -0.2, 0.2, 0.9, 0.1, -0.1, 0.2, 1.3;
    Mat9 H = Mat9::Identity();
    // Dense symmetric coupling exercises the full vec convention.
    Eigen::Matrix<double, 9, 1> v;
    v << 1, -2, 3, 1, 2, -1, 3, -2, 1;
    H += v * v.transpose();
    const auto energy = [&](const Mat3& f) {
        const Mat3 e = f * Fp.inverse();
        const Eigen::Map<const Eigen::Matrix<double, 9, 1>> x(e.data());
        return Fp.determinant() * 0.5 * x.dot(H * x);
    };
    const auto gradient = [&](const Mat3& f) {
        const MultiplicativeKinematics k(f, Fp);
        const Eigen::Matrix<double, 9, 1> p =
            H * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(k.elastic_gradient().data());
        return k.reference_gradient_F(Eigen::Map<const Mat3>(p.data()));
    };
    const MultiplicativeKinematics k(F, Fp);
    EXPECT_NEAR(k.plastic_volume_ratio(), Fp.determinant(), 1e-14);
    EXPECT_TRUE(k.elastic_gradient().isApprox(F * Fp.inverse()));
    const auto x = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(k.elastic_gradient().data());
    EXPECT_NEAR(k.reference_energy(0.5 * x.dot(H * x)), energy(F), 1e-12);
    const Mat3 P = gradient(F);
    const Mat9 HF = k.reference_hessian_F(H);
    for (double h : {1e-3, 1e-4, 1e-5}) {
        for (int j = 0; j < 9; ++j) {
            Mat3 plus = F, minus = F;
            plus.data()[j] += h;
            minus.data()[j] -= h;
            EXPECT_NEAR(P.data()[j], (energy(plus) - energy(minus)) / (2 * h), 1e-8);
            const Mat3 dP = (gradient(plus) - gradient(minus)) / (2 * h);
            EXPECT_LT((HF.col(j) - Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dP.data())).norm(), 1e-8);
        }
    }
}

TEST(MultiplicativeKinematics, ValidatesPlasticButLeavesElasticDomainToMaterial) {
    const Mat3 I = Mat3::Identity();
    Mat3 bad = I;
    bad(0, 0) = 0;
    EXPECT_THROW(MultiplicativeKinematics(I, bad), InfeasibleState);
    bad(0, 0) = -1;
    EXPECT_THROW(MultiplicativeKinematics(I, bad), InfeasibleState);
    EXPECT_NO_THROW(MultiplicativeKinematics(bad, I));
    EXPECT_NO_THROW(MultiplicativeKinematics(Mat3::Zero(), I));
    for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        bad(0, 0) = value;
        EXPECT_THROW(MultiplicativeKinematics(I, bad), std::invalid_argument);
        EXPECT_THROW(MultiplicativeKinematics(bad, I), std::invalid_argument);
    }
    bad = 1e200 * I;
    EXPECT_THROW(MultiplicativeKinematics(I, bad), InfeasibleState);
}

} // namespace
} // namespace litepgo
