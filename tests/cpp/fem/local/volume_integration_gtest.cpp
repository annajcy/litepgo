#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "test_material_sources.hpp"

#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tensor_product_quadrature_3d.hpp"
#include "quadratic_parameter_material.hpp"

#include <Eigen/LU>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
using namespace litepgo::test;

const auto element_energy = [](const Field3D& X_field, const LocalEnergy3D& energy, Eigen::VectorXd* g = nullptr,
                               Eigen::MatrixXd* H = nullptr) {
    if (g)
        g->setZero(energy.num_dofs());
    if (H)
        H->setZero(energy.num_dofs(), energy.num_dofs());
    using Q = GaussLegendre3D<2>;
    double result = 0;
    for (int q = 0; q < Q::num_points; ++q) {
        const Eigen::Vector3d xi = Q::points().row(q).transpose();
        const auto derivative = X_field.eval_dvalue_dxi(xi);
        if (derivative.rows() != 3 || derivative.cols() != 3 || !derivative.allFinite())
            throw std::invalid_argument("Invalid coordinate derivative in test fixture");
        const double determinant = Eigen::Matrix3d(derivative).determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("Invalid coordinate determinant in test fixture");
        result += energy.accumulate(xi, Q::weights()[q] * determinant, g, H);
    }
    return result;
};
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

auto eval_fields(const NodalCoefficients& coefficients, const InterpolatedField3D<HexQ1Basis>& X_field) {
    return MaterialSources{OwnedDeformationField(InterpolatedField3D<litepgo::HexQ1Basis>(coefficients.u), X_field),
                           OwnedDeformationField(InterpolatedField3D<litepgo::HexQ1Basis>(coefficients.up), X_field),
                           InterpolatedField3D<litepgo::HexQ1Basis>(coefficients.theta)};
}

template <class Member, class BindEnergy>
void check_fd(const InterpolatedField3D<HexQ1Basis>& X_field, const NodalCoefficients& f, Member member,
              BindEnergy bind_energy) {
    const auto integrate_at = [&](const auto& coefficients, Eigen::VectorXd* g = nullptr,
                                  Eigen::MatrixXd* h = nullptr) {
        return element_energy(X_field, bind_energy(eval_fields(coefficients, X_field)), g, h);
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
    const InterpolatedField3D<HexQ1Basis> X_field(coordinates());
    for (int seed = 0; seed < 3; ++seed) {
        SCOPED_TRACE(seed);
        const auto f = fields(seed);
        EXPECT_DOUBLE_EQ(
            element_energy(X_field, bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(eval_fields(f, X_field))),
            element_energy(X_field, bind_energy<MaterialLocalEnergyFp3D<NeoHookean>>(eval_fields(f, X_field))));
        EXPECT_DOUBLE_EQ(
            element_energy(X_field, bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(eval_fields(f, X_field))),
            element_energy(X_field, bind_energy<MaterialLocalEnergyTheta3D<NeoHookean>>(eval_fields(f, X_field))));
        check_fd(X_field, f, &NodalCoefficients::u, [&](const auto& local_fields) {
            return bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(local_fields);
        });
        check_fd(X_field, f, &NodalCoefficients::up, [&](const auto& local_fields) {
            return bind_energy<MaterialLocalEnergyFp3D<NeoHookean>>(local_fields);
        });
        check_fd(X_field, f, &NodalCoefficients::theta, [&](const auto& local_fields) {
            return bind_energy<MaterialLocalEnergyTheta3D<NeoHookean>>(local_fields);
        });
    }
}

TEST(VolumeIntegration, AffineFieldsHaveExpectedVolumeWeightedEnergyAndTranslationNullModes) {
    const auto X_nodes = coordinates(false);
    const InterpolatedField3D<HexQ1Basis> X_field(X_nodes);
    Eigen::Matrix3d A, P;
    A << 0.1, 0.02, -0.03, 0.04, -0.05, 0.02, 0.01, 0.03, 0.08;
    P << 0.2, 0.03, 0, 0.01, 0.1, 0.02, 0, 0.01, -0.05;
    NodalCoefficients f;
    f.u = X_nodes * A.transpose();
    f.up = X_nodes * P.transpose();
    f.theta.col(0).setConstant(2);
    f.theta.col(1).setConstant(3);
    const auto local_fields = eval_fields(f, X_field);
    Eigen::Matrix3d dX_dxi;
    dX_dxi.col(0) = X_nodes.row(1) - X_nodes.row(0);
    dX_dxi.col(1) = X_nodes.row(3) - X_nodes.row(0);
    dX_dxi.col(2) = X_nodes.row(4) - X_nodes.row(0);
    const double expected =
        dX_dxi.determinant() *
        NeoHookean::eval_W({Eigen::Matrix3d::Identity() + A, Eigen::Matrix3d::Identity() + P, Eigen::Vector2d(2, 3)});
    EXPECT_NEAR(element_energy(X_field, bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(eval_fields(f, X_field))),
                expected, 1e-12);
    Eigen::MatrixXd H, Hp;
    element_energy(X_field, bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(local_fields), nullptr, &H);
    element_energy(X_field, bind_energy<MaterialLocalEnergyFp3D<NeoHookean>>(local_fields), nullptr, &Hp);
    for (int c = 0; c < 3; ++c) {
        Eigen::Matrix<double, 24, 1> translation = Eigen::Matrix<double, 24, 1>::Zero();
        for (int i = 0; i < 8; ++i)
            translation[3 * i + c] = 1;
        EXPECT_LT((H * translation).norm(), 1e-12);
        EXPECT_LT((Hp * translation).norm(), 1e-12);
        auto shifted = f;
        shifted.u.col(c).array() += 0.7;
        shifted.up.col(c).array() -= 0.4;
        EXPECT_NEAR(
            element_energy(X_field, bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(eval_fields(shifted, X_field))),
            expected, 1e-12);
    }
}

TEST(VolumeIntegration, NonzeroParameterHessian) {
    const InterpolatedField3D<HexQ1Basis> X_field(coordinates());
    const auto coefficients = fields();
    const auto local_fields = eval_fields(coefficients, X_field);
    const auto energy = bind_energy<MaterialLocalEnergyTheta3D<QuadraticParameterMaterial<2>>>(local_fields);
    Eigen::MatrixXd H;
    element_energy(X_field, energy, nullptr, &H);
    EXPECT_GT(H.norm(), 0.1);
    check_fd(X_field, coefficients, &NodalCoefficients::theta, [&](const auto& local_fields) {
        return bind_energy<MaterialLocalEnergyTheta3D<QuadraticParameterMaterial<2>>>(local_fields);
    });
}

TEST(VolumeIntegration, PropagatesInvalidFieldsAndInfeasibleStates) {
    const InterpolatedField3D<HexQ1Basis> X_field(coordinates(false));
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
            EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field))), std::invalid_argument);
            EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field)), &g), std::invalid_argument);
            EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field)), nullptr, &h),
                         std::invalid_argument);
        }
        auto bad = fields();
        bad.theta.col(1).setConstant(-1);
        EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field))), std::invalid_argument);
        EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field)), &g), std::invalid_argument);
        EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field)), nullptr, &h),
                     std::invalid_argument);
        for (bool plastic : {false, true}) {
            bad = fields();
            auto& C = plastic ? bad.up : bad.u;
            C.setZero();
            C.col(0) = -2 * coordinates(false).col(0);
            EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field))), InfeasibleState);
            EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field)), &g), InfeasibleState);
            EXPECT_THROW(element_energy(X_field, bind_energy(eval_fields(bad, X_field)), nullptr, &h), InfeasibleState);
        }
    };
    check([&](const auto& local_fields) { return bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(local_fields); });
    check([&](const auto& local_fields) { return bind_energy<MaterialLocalEnergyFp3D<NeoHookean>>(local_fields); });
    check([&](const auto& local_fields) { return bind_energy<MaterialLocalEnergyTheta3D<NeoHookean>>(local_fields); });
}

} // namespace
