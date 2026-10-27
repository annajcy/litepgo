#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "test_material_sources.hpp"

#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/tensor_product_quadrature.hpp"
#include "quadratic_parameter_material.hpp"

#include "litepgo/fem/kinematics/reference_mapping.hpp"
#include <Eigen/LU>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
using namespace litepgo::test;
// Nodal coefficients for finite-difference perturbations in these tests only.
struct NodalCoefficients {
    Eigen::Matrix<double, 8, 3> u;
    Eigen::Matrix<double, 8, 3> up;
    Eigen::Matrix<double, 8, 2> theta;
};

Eigen::Matrix<double, 8, 3> coordinates(bool warped = true) {
    Eigen::Matrix<double, 8, 3> X_nodes;
    Eigen::Matrix3d affine_dX_dxi;
    affine_dX_dxi << 1.7, 0.2, 0.1, 0.1, 1.2, 0.2, 0.0, 0.1, 0.9;
    for (int i = 0; i < 8; ++i) {
        const auto xi = HexQ1Basis::reference_node(i);
        Eigen::Vector3d X = affine_dX_dxi * xi;
        if (warped)
            X += Eigen::Vector3d(0.15 * xi.y() * xi.z(), 0.1 * xi.x() * xi.z(), 0.05 * xi.x() * xi.y());
        X_nodes.row(i) = X.transpose();
    }
    return X_nodes;
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

auto evaluate_fields(const NodalCoefficients& coefficients, const InterpolatedField<HexQ1Basis>& X_field) {
    return MaterialSources{DeformationGradient(InterpolatedField<litepgo::HexQ1Basis>(coefficients.u), X_field),
                           DeformationGradient(InterpolatedField<litepgo::HexQ1Basis>(coefficients.up), X_field),
                           InterpolatedField<litepgo::HexQ1Basis>(coefficients.theta)};
}

template <class Member, class BindEnergy>
void check_fd(const InterpolatedField<HexQ1Basis>& X_field, const NodalCoefficients& f, Member member,
              BindEnergy bind_energy) {
    const auto integrate_at = [&](const auto& coefficients, Eigen::VectorXd* g = nullptr,
                                  Eigen::MatrixXd* h = nullptr) {
        return integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(coefficients, X_field)), g, h);
    };
    Eigen::VectorXd G;
    Eigen::MatrixXd H;
    integrate_at(f, &G, &H);
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
            Eigen::VectorXd plus_g, minus_g;
            fdG[j] = (integrate_at(plus, &plus_g) - integrate_at(minus, &minus_g)) / (2 * h);
            fdH.col(j) = (plus_g - minus_g) / (2 * h);
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
    const double fd = (integrate_at(pp) - integrate_at(pm) - integrate_at(mp) + integrate_at(mm)) / (4 * h * h);
    EXPECT_NEAR(fd, d.dot(H * e), 2e-6 * (1 + H.norm()));
}

TEST(VolumeIntegration, HexQ1LocalDofFiniteDifferencesForAllThreeFields) {
    const InterpolatedField<HexQ1Basis> X_field(coordinates());
    for (int seed = 0; seed < 3; ++seed) {
        SCOPED_TRACE(seed);
        const auto f = fields(seed);
        EXPECT_DOUBLE_EQ(integrate_volume<GaussLegendre3D<2>>(
                             bind_energy<NeoHookean, MaterialVariable::F>(evaluate_fields(f, X_field))),
                         integrate_volume<GaussLegendre3D<2>>(
                             bind_energy<NeoHookean, MaterialVariable::Fp>(evaluate_fields(f, X_field))));
        EXPECT_DOUBLE_EQ(integrate_volume<GaussLegendre3D<2>>(
                             bind_energy<NeoHookean, MaterialVariable::F>(evaluate_fields(f, X_field))),
                         integrate_volume<GaussLegendre3D<2>>(
                             bind_energy<NeoHookean, MaterialVariable::Theta>(evaluate_fields(f, X_field))));
        check_fd(X_field, f, &NodalCoefficients::u,
                 [&](const auto& local_fields) { return bind_energy<NeoHookean, MaterialVariable::F>(local_fields); });
        check_fd(X_field, f, &NodalCoefficients::up,
                 [&](const auto& local_fields) { return bind_energy<NeoHookean, MaterialVariable::Fp>(local_fields); });
        check_fd(X_field, f, &NodalCoefficients::theta, [&](const auto& local_fields) {
            return bind_energy<NeoHookean, MaterialVariable::Theta>(local_fields);
        });
    }
}

TEST(VolumeIntegration, AffineFieldsHaveExpectedVolumeWeightedEnergyAndTranslationNullModes) {
    const auto X_nodes = coordinates(false);
    const InterpolatedField<HexQ1Basis> X_field(X_nodes);
    Eigen::Matrix3d A, P;
    A << 0.1, 0.02, -0.03, 0.04, -0.05, 0.02, 0.01, 0.03, 0.08;
    P << 0.2, 0.03, 0, 0.01, 0.1, 0.02, 0, 0.01, -0.05;
    NodalCoefficients f;
    f.u = X_nodes * A.transpose();
    f.up = X_nodes * P.transpose();
    f.theta.col(0).setConstant(2);
    f.theta.col(1).setConstant(3);
    const auto local_fields = evaluate_fields(f, X_field);
    Eigen::Matrix3d dX_dxi;
    dX_dxi.col(0) = X_nodes.row(1) - X_nodes.row(0);
    dX_dxi.col(1) = X_nodes.row(3) - X_nodes.row(0);
    dX_dxi.col(2) = X_nodes.row(4) - X_nodes.row(0);
    const double expected =
        dX_dxi.determinant() *
        NeoHookean::value({Eigen::Matrix3d::Identity() + A, Eigen::Matrix3d::Identity() + P, Eigen::Vector2d(2, 3)});
    EXPECT_NEAR(
        integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::F>(evaluate_fields(f, X_field))),
        expected, 1e-12);
    Eigen::MatrixXd H, Hp;
    integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::F>(local_fields), nullptr, &H);
    integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Fp>(local_fields), nullptr, &Hp);
    for (int c = 0; c < 3; ++c) {
        Eigen::Matrix<double, 24, 1> translation = Eigen::Matrix<double, 24, 1>::Zero();
        for (int i = 0; i < 8; ++i)
            translation[3 * i + c] = 1;
        EXPECT_LT((H * translation).norm(), 1e-12);
        EXPECT_LT((Hp * translation).norm(), 1e-12);
        auto shifted = f;
        shifted.u.col(c).array() += 0.7;
        shifted.up.col(c).array() -= 0.4;
        EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(
                        bind_energy<NeoHookean, MaterialVariable::F>(evaluate_fields(shifted, X_field))),
                    expected, 1e-12);
    }
}

TEST(VolumeIntegration, NonzeroParameterHessian) {
    const InterpolatedField<HexQ1Basis> X_field(coordinates());
    const auto coefficients = fields();
    const auto energy =
        bind_energy<QuadraticParameterMaterial<2>, MaterialVariable::Theta>(evaluate_fields(coefficients, X_field));
    Eigen::MatrixXd H;
    integrate_volume<GaussLegendre3D<2>>(energy, nullptr, &H);
    EXPECT_GT(H.norm(), 0.1);
    check_fd(X_field, coefficients, &NodalCoefficients::theta, [&](const auto& local_fields) {
        return bind_energy<QuadraticParameterMaterial<2>, MaterialVariable::Theta>(local_fields);
    });
}

TEST(VolumeIntegration, PropagatesInvalidFieldsAndInfeasibleStates) {
    const InterpolatedField<HexQ1Basis> X_field(coordinates(false));
    const auto check = [&](auto bind_energy) {
        Eigen::VectorXd g;
        Eigen::MatrixXd h;
        for (int which = 0; which < 3; ++which) {
            auto bad = fields();
            if (which == 0)
                bad.u(0, 0) = std::numeric_limits<double>::quiet_NaN();
            if (which == 1)
                bad.up(0, 0) = std::numeric_limits<double>::infinity();
            if (which == 2)
                bad.theta(0, 0) = std::numeric_limits<double>::quiet_NaN();
            EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field))),
                         std::invalid_argument);
            EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field)), &g),
                         std::invalid_argument);
            EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field)), nullptr, &h),
                         std::invalid_argument);
        }
        auto bad = fields();
        bad.theta.col(1).setConstant(-1);
        EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field))),
                     std::invalid_argument);
        EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field)), &g),
                     std::invalid_argument);
        EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field)), nullptr, &h),
                     std::invalid_argument);
        for (bool plastic : {false, true}) {
            bad = fields();
            auto& C = plastic ? bad.up : bad.u;
            C.setZero();
            C.col(0) = -2 * coordinates(false).col(0);
            EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field))),
                         InfeasibleState);
            EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field)), &g),
                         InfeasibleState);
            EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy(evaluate_fields(bad, X_field)), nullptr, &h),
                         InfeasibleState);
        }
    };
    check([&](const auto& local_fields) { return bind_energy<NeoHookean, MaterialVariable::F>(local_fields); });
    check([&](const auto& local_fields) { return bind_energy<NeoHookean, MaterialVariable::Fp>(local_fields); });
    check([&](const auto& local_fields) { return bind_energy<NeoHookean, MaterialVariable::Theta>(local_fields); });
}

} // namespace
