#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/interpolated_value_field.hpp"
#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/geometry/volume_quadrature_cache.hpp"
#include "litepgo/fem/local/local_kernel.hpp"
#include "litepgo/fem/local/neo_hookean_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/tensor_product_quadrature.hpp"
#include "quadratic_parameter_local_energy.hpp"

#include "litepgo/fem/geometry/volume_reference_geometry.hpp"
#include <Eigen/LU>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
// Nodal coefficients for finite-difference perturbations in these tests only.
struct NodalCoefficients {
    Eigen::Matrix<double, 8, 3> u;
    Eigen::Matrix<double, 8, 3> up;
    Eigen::Matrix<double, 8, 2> theta;
};

Eigen::Matrix<double, 8, 3> coordinates(bool warped = true) {
    Eigen::Matrix<double, 8, 3> X;
    Eigen::Matrix3d J;
    J << 1.7, 0.2, 0.1, 0.1, 1.2, 0.2, 0.0, 0.1, 0.9;
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        Eigen::Vector3d x = J * xi;
        if (warped)
            x += Eigen::Vector3d(0.15 * xi.y() * xi.z(), 0.1 * xi.x() * xi.z(), 0.05 * xi.x() * xi.y());
        X.row(i) = x.transpose();
    }
    return X;
}

NodalCoefficients fields(int seed = 0) {
    NodalCoefficients f;
    for (int i = 0; i < 8; ++i) {
        for (int c = 0; c < 3; ++c) {
            f.u(i, c) = 0.06 * std::sin(0.7 * (i * 3 + c + 1) * (seed + 1));
            f.up(i, c) = 0.035 * std::cos(0.4 * (i * 3 + c + 1) * (seed + 2));
        }
        f.theta(i, 0) = 2 + 0.15 * i;
        f.theta(i, 1) = 3 + 0.1 * std::sin(i + seed);
    }
    return f;
}

auto evaluate_fields(const NodalCoefficients& coefficients) {
    return MaterialFields{std::make_shared<DeformationGradientField>(coefficients.u),
                          std::make_shared<DeformationGradientField>(coefficients.up),
                          std::make_shared<InterpolatedValueField>(coefficients.theta)};
}

template <class Kernel, class Member, class BindEnergy>
void check_fd(const Kernel& kernel, const NodalCoefficients& f, Member member, BindEnergy bind_energy) {
    const auto gradient_at = [&](const auto& coefficients) {
        const auto local_fields = evaluate_fields(coefficients);
        return kernel.gradient(bind_energy(local_fields));
    };
    const auto local_fields = evaluate_fields(f);
    const auto G = gradient_at(f);
    const auto H = kernel.hessian(bind_energy(local_fields));
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    const int components = static_cast<int>((f.*member).cols());
    for (double h : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(h);
        auto fdG = G;
        auto fdH = H;
        for (int j = 0; j < G.size(); ++j) {
            auto plus = f, minus = f;
            (plus.*member)(j / components, j % components) += h;
            (minus.*member)(j / components, j % components) -= h;
            fdG[j] =
                (kernel.value(bind_energy(evaluate_fields(plus))) - kernel.value(bind_energy(evaluate_fields(minus)))) /
                (2 * h);
            fdH.col(j) = (gradient_at(plus) - gradient_at(minus)) / (2 * h);
        }
        EXPECT_LT((G - fdG).norm() / (1 + G.norm()), 20 * h * h + 1e-9);
        EXPECT_LT((H - fdH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
    }
    // Independent mixed directional second difference of the scalar energy.
    auto d = G, e = G;
    for (int j = 0; j < G.size(); ++j) {
        d[j] = std::sin(j + 0.3);
        e[j] = std::cos(0.7 * j + 0.2);
    }
    d.normalize();
    e.normalize();
    constexpr double h = 3e-4;
    auto pp = f, pm = f, mp = f, mm = f;
    for (int j = 0; j < G.size(); ++j) {
        const int row = j / components, col = j % components;
        (pp.*member)(row, col) += h * (d[j] + e[j]);
        (pm.*member)(row, col) += h * (d[j] - e[j]);
        (mp.*member)(row, col) += h * (-d[j] + e[j]);
        (mm.*member)(row, col) -= h * (d[j] + e[j]);
    }
    const double fd =
        (kernel.value(bind_energy(evaluate_fields(pp))) - kernel.value(bind_energy(evaluate_fields(pm))) -
         kernel.value(bind_energy(evaluate_fields(mp))) + kernel.value(bind_energy(evaluate_fields(mm)))) /
        (4 * h * h);
    EXPECT_NEAR(fd, d.dot(H * e), 2e-6 * (1 + H.norm()));
}

TEST(VolumeIntegration, HexQ1LocalDofFiniteDifferencesForAllThreeFields) {
    const litepgo::VolumeQuadratureCache cache{VolumeReferenceGeometry<HexQ1Basis>(coordinates()),
                                               litepgo::GaussLegendre3D<2>{}};
    const litepgo::LocalKernel kernel(cache);
    for (int seed = 0; seed < 3; ++seed) {
        SCOPED_TRACE(seed);
        const auto f = fields(seed);
        EXPECT_DOUBLE_EQ(kernel.value(NeoHookeanLocalEnergyF(evaluate_fields(f))),
                         kernel.value(NeoHookeanLocalEnergyFp(evaluate_fields(f))));
        EXPECT_DOUBLE_EQ(kernel.value(NeoHookeanLocalEnergyF(evaluate_fields(f))),
                         kernel.value(NeoHookeanLocalEnergyTheta(evaluate_fields(f))));
        check_fd(kernel, f, &NodalCoefficients::u,
                 [](const auto& local_fields) { return NeoHookeanLocalEnergyF(local_fields); });
        check_fd(kernel, f, &NodalCoefficients::up,
                 [](const auto& local_fields) { return NeoHookeanLocalEnergyFp(local_fields); });
        check_fd(kernel, f, &NodalCoefficients::theta,
                 [](const auto& local_fields) { return NeoHookeanLocalEnergyTheta(local_fields); });
    }
}

TEST(VolumeIntegration, AffineFieldsHaveExpectedVolumeWeightedEnergyAndTranslationNullModes) {
    const auto X = coordinates(false);
    const litepgo::VolumeQuadratureCache cache{VolumeReferenceGeometry<HexQ1Basis>(X), litepgo::GaussLegendre3D<2>{}};
    const litepgo::LocalKernel kernel(cache);
    Eigen::Matrix3d A, P;
    A << 0.1, 0.02, -0.03, 0.04, -0.05, 0.02, 0.01, 0.03, 0.08;
    P << 0.2, 0.03, 0, 0.01, 0.1, 0.02, 0, 0.01, -0.05;
    NodalCoefficients f;
    f.u = X * A.transpose();
    f.up = X * P.transpose();
    f.theta.col(0).setConstant(2);
    f.theta.col(1).setConstant(3);
    const auto local_fields = evaluate_fields(f);
    Eigen::Matrix3d J;
    J.col(0) = X.row(1) - X.row(0);
    J.col(1) = X.row(3) - X.row(0);
    J.col(2) = X.row(4) - X.row(0);
    const double expected =
        J.determinant() *
        NeoHookean{}.value({Eigen::Matrix3d::Identity() + A, Eigen::Matrix3d::Identity() + P, Eigen::Vector2d(2, 3)});
    EXPECT_NEAR(kernel.value(NeoHookeanLocalEnergyF(evaluate_fields(f))), expected, 1e-12);
    for (int c = 0; c < 3; ++c) {
        Eigen::Matrix<double, 24, 1> translation = Eigen::Matrix<double, 24, 1>::Zero();
        for (int i = 0; i < 8; ++i)
            translation[3 * i + c] = 1;
        EXPECT_LT((kernel.hessian(NeoHookeanLocalEnergyF(local_fields)) * translation).norm(), 1e-12);
        EXPECT_LT((kernel.hessian(NeoHookeanLocalEnergyFp(local_fields)) * translation).norm(), 1e-12);
        auto shifted = f;
        shifted.u.col(c).array() += 0.7;
        shifted.up.col(c).array() -= 0.4;
        EXPECT_NEAR(kernel.value(NeoHookeanLocalEnergyF(evaluate_fields(shifted))), expected, 1e-12);
    }
}

TEST(VolumeIntegration, NonzeroParameterHessian) {
    const litepgo::VolumeQuadratureCache cache{VolumeReferenceGeometry<HexQ1Basis>(coordinates()),
                                               litepgo::GaussLegendre3D<2>{}};
    const LocalKernel kernel(cache);
    const auto coefficients = fields();
    const QuadraticParameterLocalEnergy energy(evaluate_fields(coefficients));
    EXPECT_GT(kernel.hessian(energy).norm(), 0.1);
    check_fd(kernel, coefficients, &NodalCoefficients::theta,
             [](const auto& local_fields) { return QuadraticParameterLocalEnergy(local_fields); });
}

TEST(VolumeIntegration, PropagatesInvalidFieldsAndInfeasibleStates) {
    const litepgo::VolumeQuadratureCache cache{VolumeReferenceGeometry<HexQ1Basis>(coordinates(false)),
                                               litepgo::GaussLegendre3D<2>{}};
    const litepgo::LocalKernel kernel(cache);
    const auto check = [&](const auto& kernel, auto bind_energy) {
        for (int which = 0; which < 3; ++which) {
            auto bad = fields();
            if (which == 0)
                bad.u(0, 0) = std::numeric_limits<double>::quiet_NaN();
            if (which == 1)
                bad.up(0, 0) = std::numeric_limits<double>::infinity();
            if (which == 2)
                bad.theta(0, 0) = std::numeric_limits<double>::quiet_NaN();
            EXPECT_THROW(kernel.value(bind_energy(evaluate_fields(bad))), std::invalid_argument);
            EXPECT_THROW(kernel.gradient(bind_energy(evaluate_fields(bad))), std::invalid_argument);
            EXPECT_THROW(kernel.hessian(bind_energy(evaluate_fields(bad))), std::invalid_argument);
        }
        auto bad = fields();
        bad.theta.col(1).setConstant(-1);
        EXPECT_THROW(kernel.value(bind_energy(evaluate_fields(bad))), std::invalid_argument);
        EXPECT_THROW(kernel.gradient(bind_energy(evaluate_fields(bad))), std::invalid_argument);
        EXPECT_THROW(kernel.hessian(bind_energy(evaluate_fields(bad))), std::invalid_argument);
        for (bool plastic : {false, true}) {
            bad = fields();
            auto& C = plastic ? bad.up : bad.u;
            C.setZero();
            C.col(0) = -2 * coordinates(false).col(0);
            EXPECT_THROW(kernel.value(bind_energy(evaluate_fields(bad))), InfeasibleState);
            EXPECT_THROW(kernel.gradient(bind_energy(evaluate_fields(bad))), InfeasibleState);
            EXPECT_THROW(kernel.hessian(bind_energy(evaluate_fields(bad))), InfeasibleState);
        }
    };
    check(kernel, [](const auto& local_fields) { return NeoHookeanLocalEnergyF(local_fields); });
    check(kernel, [](const auto& local_fields) { return NeoHookeanLocalEnergyFp(local_fields); });
    check(kernel, [](const auto& local_fields) { return NeoHookeanLocalEnergyTheta(local_fields); });
}

} // namespace
