#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "test_material_sources.hpp"

#include "litepgo/fem/local/integrate_volume.hpp"
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

auto fields(const Coefficients& f, const InterpolatedField<TetP1Basis>& X_field) {
    return MaterialSources{DeformationGradient(InterpolatedField<TetP1Basis>(f.u), X_field),
                           DeformationGradient(InterpolatedField<TetP1Basis>(f.up), X_field),
                           InterpolatedField<TetP1Basis>(f.theta)};
}

template <class Member, class BindEnergy> void check_fd(Member member, BindEnergy bind_energy) {
    const InterpolatedField<TetP1Basis> X_field(coordinates());
    const auto f = coefficients();
    const auto evaluate = [&](const Coefficients& current, Eigen::VectorXd* g = nullptr, Eigen::MatrixXd* H = nullptr) {
        return integrate_volume<TetCentroidQuadrature>(bind_energy(fields(current, X_field)), g, H);
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
    const InterpolatedField<TetP1Basis> X_field(X_nodes);
    const auto f = coefficients();
    const auto local_fields = fields(f, X_field);
    const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + edge_matrix(f.u) * Dm.inverse();
    const Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity() + edge_matrix(f.up) * Dm.inverse();
    const std::array<Eigen::Vector3d, 3> samples{
        {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0.2, 0.3, 0.1), Eigen::Vector3d(0.25, 0.25, 0.25)}};
    for (const auto& xi : samples) {
        const auto point = litepgo::ReferenceMapping(X_field.evaluate(xi));
        EXPECT_TRUE(point.X().isApprox(X_nodes.row(0).transpose() + Dm * xi, 1e-14));
        EXPECT_TRUE(point.dX_dxi().isApprox(Dm, 1e-14));
        EXPECT_TRUE(point.dxi_dX().isApprox(Dm.inverse(), 1e-14));
        EXPECT_NEAR(point.det_dX_dxi(), Dm.determinant(), 1e-14);
        EXPECT_TRUE(local_fields.F.evaluate(point.xi()).F().isApprox(F, 1e-14));
        EXPECT_TRUE(local_fields.Fp.evaluate(point.xi()).F().isApprox(Fp, 1e-14));
    }
    // F and Fp are constant, and NeoHookean is linear in the P1 parameter field.
    const double expected = Dm.determinant() / 6 * NeoHookean::value({F, Fp, f.theta.colwise().mean().transpose()});
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    EXPECT_NEAR(
        integrate_volume<TetCentroidQuadrature>(bind_energy<NeoHookean, MaterialVariable::F>(local_fields), &g, &H),
        expected, 1e-13);
    EXPECT_NEAR(integrate_volume<TetCentroidQuadrature>(bind_energy<NeoHookean, MaterialVariable::Fp>(local_fields)),
                expected, 1e-13);
    EXPECT_NEAR(integrate_volume<TetCentroidQuadrature>(bind_energy<NeoHookean, MaterialVariable::Theta>(local_fields)),
                expected, 1e-13);
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
        check_fd(&Coefficients::u, [](auto f) { return bind_energy<NeoHookean, MaterialVariable::F>(std::move(f)); });
    }
    {
        SCOPED_TRACE("Fp");
        check_fd(&Coefficients::up, [](auto f) { return bind_energy<NeoHookean, MaterialVariable::Fp>(std::move(f)); });
    }
    {
        SCOPED_TRACE("theta");
        check_fd(&Coefficients::theta,
                 [](auto f) { return bind_energy<NeoHookean, MaterialVariable::Theta>(std::move(f)); });
    }
}

TEST(TetP1Integration, RejectsDegenerateAndNegativelyOrientedRestGeometry) {
    for (bool degenerate : {false, true}) {
        auto X_nodes = coordinates();
        if (degenerate)
            X_nodes.row(3) = X_nodes.row(0);
        else
            X_nodes.row(1).swap(X_nodes.row(2));
        const InterpolatedField<TetP1Basis> X_field(X_nodes);
        EXPECT_THROW(integrate_volume<TetCentroidQuadrature>(
                         bind_energy<NeoHookean, MaterialVariable::F>(fields(coefficients(), X_field))),
                     std::invalid_argument);
    }
}
