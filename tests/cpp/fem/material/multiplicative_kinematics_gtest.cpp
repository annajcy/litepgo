#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/material/multiplicative_kinematics.hpp"

#include <Eigen/LU>
#include <gtest/gtest.h>
#include <limits>

namespace litepgo {
namespace {

TEST(MultiplicativeKinematics, IdentityAndOwnedSnapshot) {
    Eigen::Matrix3d F = Eigen::Matrix3d::Random();
    const Eigen::Matrix3d original = F;
    MultiplicativeKinematics k(F, Eigen::Matrix3d::Identity());
    F.setZero();
    EXPECT_TRUE(k.elastic_gradient().isApprox(original));
    EXPECT_DOUBLE_EQ(k.plastic_volume_ratio(), 1.0);
    EXPECT_DOUBLE_EQ(k.eval_W(2.5), 2.5);
    EXPECT_TRUE(k.eval_dW_dF(original).isApprox(original));
    const Eigen::Matrix<double, 9, 9> H = Eigen::Matrix<double, 9, 9>::Random();
    EXPECT_TRUE(k.eval_d2W_dF2(H).isApprox(H));
}

TEST(MultiplicativeKinematics, PullbackMatchesFiniteDifferences) {
    Eigen::Matrix3d Fp;
    Fp << 1.4, 0.2, -0.1, 0.1, 0.8, 0.3, 0.0, -0.2, 1.2;
    Eigen::Matrix3d F;
    F << 1.1, 0.3, -0.2, 0.2, 0.9, 0.1, -0.1, 0.2, 1.3;
    Eigen::Matrix<double, 9, 9> H = Eigen::Matrix<double, 9, 9>::Identity();
    // Dense symmetric coupling exercises the full vec convention.
    Eigen::Matrix<double, 9, 1> v;
    v << 1, -2, 3, 1, 2, -1, 3, -2, 1;
    H += v * v.transpose();
    const auto energy = [&](const Eigen::Matrix3d& f) {
        const Eigen::Matrix3d e = f * Fp.inverse();
        const Eigen::Map<const Eigen::Matrix<double, 9, 1>> x(e.data());
        return Fp.determinant() * 0.5 * x.dot(H * x);
    };
    const auto gradient = [&](const Eigen::Matrix3d& f) {
        const MultiplicativeKinematics k(f, Fp);
        const Eigen::Matrix<double, 9, 1> p =
            H * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(k.elastic_gradient().data());
        return k.eval_dW_dF(Eigen::Map<const Eigen::Matrix3d>(p.data()));
    };
    const MultiplicativeKinematics k(F, Fp);
    EXPECT_NEAR(k.plastic_volume_ratio(), Fp.determinant(), 1e-14);
    EXPECT_TRUE(k.elastic_gradient().isApprox(F * Fp.inverse()));
    const auto x = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(k.elastic_gradient().data());
    EXPECT_NEAR(k.eval_W(0.5 * x.dot(H * x)), energy(F), 1e-12);
    const Eigen::Matrix3d P = gradient(F);
    const Eigen::Matrix<double, 9, 9> HF = k.eval_d2W_dF2(H);
    for (double h : {1e-3, 1e-4, 1e-5}) {
        for (int j = 0; j < 9; ++j) {
            Eigen::Matrix3d plus = F, minus = F;
            plus.data()[j] += h;
            minus.data()[j] -= h;
            EXPECT_NEAR(P.data()[j], (energy(plus) - energy(minus)) / (2 * h), 1e-8);
            const Eigen::Matrix3d dP = (gradient(plus) - gradient(minus)) / (2 * h);
            EXPECT_LT((HF.col(j) - Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dP.data())).norm(), 1e-8);
        }
    }
}

TEST(MultiplicativeKinematics, PlasticPullbackMatchesFiniteDifferences) {

    Eigen::Matrix3d F, Fp;
    F << 1.1, 0.3, -0.2, 0.2, 0.9, 0.1, -0.1, 0.2, 1.3;
    Fp << 1.4, 0.2, -0.1, 0.1, 0.8, 0.3, 0.0, -0.2, 1.2;
    Eigen::Matrix<double, 9, 1> v;
    v << 1, -2, 3, 1, 2, -1, 3, -2, 1;
    const Eigen::Matrix<double, 9, 9> d2psi_dFe2 = Eigen::Matrix<double, 9, 9>::Identity() + v * v.transpose();
    // A nonzero constant isolates the determinant derivative as well.
    for (bool constant_only : {true, false}) {
        SCOPED_TRACE(constant_only);
        const Eigen::Matrix<double, 9, 9> H =
            constant_only ? Eigen::Matrix<double, 9, 9>(Eigen::Matrix<double, 9, 9>::Zero()) : d2psi_dFe2;
        const auto energy = [&](const Eigen::Matrix3d& p) {
            const Eigen::Matrix3d Fe = F * p.inverse();
            const Eigen::Map<const Eigen::Matrix<double, 9, 1>> x(Fe.data());
            return p.determinant() * (2.3 + 0.5 * x.dot(H * x));
        };
        const auto gradient = [&](const Eigen::Matrix3d& p) {
            const MultiplicativeKinematics k(F, p);
            const Eigen::Map<const Eigen::Matrix<double, 9, 1>> x(k.elastic_gradient().data());
            const Eigen::Matrix<double, 9, 1> pe = H * x;
            return k.eval_dW_dFp(2.3 + 0.5 * x.dot(pe), Eigen::Map<const Eigen::Matrix3d>(pe.data()));
        };
        const MultiplicativeKinematics k(F, Fp);
        const Eigen::Map<const Eigen::Matrix<double, 9, 1>> x(k.elastic_gradient().data());
        const Eigen::Matrix<double, 9, 1> pe = H * x;
        const Eigen::Matrix3d G = gradient(Fp);
        const Eigen::Matrix<double, 9, 9> Hp =
            k.eval_d2W_dFp2(2.3 + 0.5 * x.dot(pe), Eigen::Map<const Eigen::Matrix3d>(pe.data()), H);
        EXPECT_LT((Hp - Hp.transpose()).norm(), 1e-12 * (1 + Hp.norm()));
        for (double h : {1e-3, 3e-4, 1e-4}) {
            SCOPED_TRACE(h);
            Eigen::Matrix3d fdG;
            Eigen::Matrix<double, 9, 9> fdH;
            for (int j = 0; j < 9; ++j) {
                Eigen::Matrix3d plus = Fp, minus = Fp;
                plus.data()[j] += h;
                minus.data()[j] -= h;
                fdG.data()[j] = (energy(plus) - energy(minus)) / (2 * h);
                const Eigen::Matrix3d dG = (gradient(plus) - gradient(minus)) / (2 * h);
                fdH.col(j) = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(dG.data());
            }
            EXPECT_LT((G - fdG).norm() / (1 + G.norm()), 20 * h * h + 1e-9);
            EXPECT_LT((Hp - fdH).norm() / (1 + Hp.norm()), 20 * h * h + 1e-9);
        }
    }
}

TEST(MultiplicativeKinematics, ValidatesPlasticButLeavesElasticDomainToMaterial) {
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d bad = I;
    bad(0, 0) = 0;
    EXPECT_THROW(MultiplicativeKinematics(I, bad), InfeasibleState);
    bad(0, 0) = -1;
    EXPECT_THROW(MultiplicativeKinematics(I, bad), InfeasibleState);
    EXPECT_NO_THROW(MultiplicativeKinematics(bad, I));
    EXPECT_NO_THROW(MultiplicativeKinematics(Eigen::Matrix3d::Zero(), I));
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
