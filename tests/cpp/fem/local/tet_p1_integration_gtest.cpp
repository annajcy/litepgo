#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "test_material_sources.hpp"

#include <Eigen/LU>

#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {
using namespace litepgo;
using namespace litepgo::test;

const auto element_energy = [](const Field3D& X_field, const LocalEnergy3D& energy, Eigen::VectorXd* g = nullptr,
                               Eigen::MatrixXd* H = nullptr) {
    if (g)
        g->setZero(energy.num_dofs());
    if (H)
        H->setZero(energy.num_dofs(), energy.num_dofs());
    using Q = TetCentroidQuadrature;
    double result = 0;
    for (int q = 0; q < Q::num_points; ++q) {
        const Eigen::Vector3d xi = Q::points().row(q).transpose();
        const auto derivative = X_field.dvalue_dxi(xi);
        if (derivative.rows() != 3 || derivative.cols() != 3 || !derivative.allFinite())
            throw std::invalid_argument("Invalid coordinate derivative in test fixture");
        const double determinant = Eigen::Matrix3d(derivative).determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("Invalid coordinate determinant in test fixture");
        result += energy.accumulate(xi, Q::weights()[q] * determinant, g, H);
    }
    return result;
};

Eigen::Matrix<double, 4, 3> coordinates() {
    Eigen::Matrix<double, 4, 3> X_nodes;
    X_nodes << 0.2, -0.3, 0.4, 1.9, -0.2, 0.4, 0.4, 0.9, 0.5, 0.3, -0.1, 1.3;
    return X_nodes;
}

Eigen::Matrix3d edge_matrix(const Eigen::Matrix<double, 4, 3>& X_nodes) {
    return (X_nodes.bottomRows<3>().rowwise() - X_nodes.row(0)).transpose();
}

struct Coefficients {
    Eigen::Matrix<double, 4, 3> u, up;
    Eigen::Matrix<double, 4, 2> theta;
};

Coefficients coefficients() {
    Coefficients f;
    for (int i = 0; i < 4; ++i) {
        for (int c = 0; c < 3; ++c) {
            f.u(i, c) = 0.04 * std::sin(i * 3 + c + 0.3);
            f.up(i, c) = 0.025 * std::cos(i * 3 + c + 0.2);
        }
        f.theta.row(i) << 2 + 0.1 * i, 3 + 0.05 * i;
    }
    return f;
}

auto fields(const Coefficients& f, const InterpolatedField3D<TetP1Basis>& X_field) {
    return MaterialSources{OwnedDeformationField(InterpolatedField3D<TetP1Basis>(f.u), X_field),
                           OwnedDeformationField(InterpolatedField3D<TetP1Basis>(f.up), X_field),
                           InterpolatedField3D<TetP1Basis>(f.theta)};
}

template <class Member, class BindEnergy> void check_fd(Member member, BindEnergy bind_energy) {
    const InterpolatedField3D<TetP1Basis> X_field(coordinates());
    const auto f = coefficients();
    const auto evaluate = [&](const Coefficients& current, Eigen::VectorXd* g = nullptr, Eigen::MatrixXd* H = nullptr) {
        return element_energy(X_field, bind_energy(fields(current, X_field)), g, H);
    };
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    evaluate(f, &g, &H);
    ASSERT_EQ(g.size(), (f.*member).size());
    ASSERT_EQ(H.rows(), g.size());
    ASSERT_EQ(H.cols(), g.size());
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    const int components = static_cast<int>((f.*member).cols());
    for (double h : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(h);
        Eigen::VectorXd fd_g(g.size());
        Eigen::MatrixXd fd_h(H.rows(), H.cols());
        for (int j = 0; j < g.size(); ++j) {
            auto plus = f, minus = f;
            (plus.*member)(j / components, j % components) += h;
            (minus.*member)(j / components, j % components) -= h;
            Eigen::VectorXd plus_g, minus_g;
            fd_g[j] = (evaluate(plus, &plus_g) - evaluate(minus, &minus_g)) / (2 * h);
            fd_h.col(j) = (plus_g - minus_g) / (2 * h);
        }
        EXPECT_LT((g - fd_g).norm() / (1 + g.norm()), 10 * h * h + 1e-9);
        EXPECT_LT((H - fd_h).norm() / (1 + H.norm()), 10 * h * h + 1e-9);
    }
}
} // namespace

TEST(TetP1Integration, MatchesDirectGeometryAndDeformationGradient) {
    const auto X_nodes = coordinates();
    const Eigen::Matrix3d Dm = edge_matrix(X_nodes);
    const InterpolatedField3D<TetP1Basis> X_field(X_nodes);
    const auto f = coefficients();
    const auto local_fields = fields(f, X_field);
    const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + edge_matrix(f.u) * Dm.inverse();
    const Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity() + edge_matrix(f.up) * Dm.inverse();
    const std::array<Eigen::Vector3d, 3> samples{
        {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.2, 0.3, 0.1), Eigen::Vector3d(0.25, 0.25, 0.25)}};
    for (const auto& xi : samples) {
        EXPECT_TRUE(X_field.value(xi).isApprox(X_nodes.row(0).transpose() + Dm * xi, 1e-14));
        EXPECT_TRUE(X_field.dvalue_dxi(xi).isApprox(Dm, 1e-14));
        EXPECT_TRUE(X_field.dvalue_dxi(xi).inverse().isApprox(Dm.inverse(), 1e-14));
        EXPECT_NEAR(X_field.dvalue_dxi(xi).determinant(), Dm.determinant(), 1e-14);
        EXPECT_TRUE(local_fields.F.value(xi).reshaped(3, 3).isApprox(F, 1e-14));
        EXPECT_TRUE(local_fields.Fp.value(xi).reshaped(3, 3).isApprox(Fp, 1e-14));
    }
    // F and Fp are constant, and NeoHookean is linear in the P1 parameter field.
    const double expected = Dm.determinant() / 6 * NeoHookean::value({F, Fp, f.theta.colwise().mean().transpose()});
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    EXPECT_NEAR(element_energy(X_field, bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(local_fields), &g, &H),
                expected, 1e-13);
    EXPECT_NEAR(element_energy(X_field, bind_energy<MaterialLocalEnergyFp3D<NeoHookean>>(local_fields)), expected,
                1e-13);
    EXPECT_NEAR(element_energy(X_field, bind_energy<MaterialLocalEnergyTheta3D<NeoHookean>>(local_fields)), expected,
                1e-13);
    for (int c = 0; c < 3; ++c) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(12);
        for (int i = 0; i < 4; ++i)
            translation[3 * i + c] = 1;
        EXPECT_NEAR(g.dot(translation), 0, 1e-13);
        EXPECT_LT((H * translation).norm(), 1e-12);
    }
}

TEST(TetP1Integration, LocalDofFiniteDifferencesForAllThreeFields) {
    {
        SCOPED_TRACE("F");
        check_fd(&Coefficients::u, [](const auto& f) { return bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(f); });
    }
    {
        SCOPED_TRACE("Fp");
        check_fd(&Coefficients::up, [](const auto& f) { return bind_energy<MaterialLocalEnergyFp3D<NeoHookean>>(f); });
    }
    {
        SCOPED_TRACE("theta");
        check_fd(&Coefficients::theta,
                 [](const auto& f) { return bind_energy<MaterialLocalEnergyTheta3D<NeoHookean>>(f); });
    }
}

TEST(TetP1Integration, RejectsDegenerateAndNegativelyOrientedRestGeometry) {
    for (bool degenerate : {false, true}) {
        auto X_nodes = coordinates();
        if (degenerate)
            X_nodes.row(3) = X_nodes.row(0);
        else
            X_nodes.row(1).swap(X_nodes.row(2));
        const InterpolatedField3D<TetP1Basis> X_field(X_nodes);
        EXPECT_THROW(
            element_energy(X_field, bind_energy<MaterialLocalEnergyF3D<NeoHookean>>(fields(coefficients(), X_field))),
            std::invalid_argument);
    }
}
