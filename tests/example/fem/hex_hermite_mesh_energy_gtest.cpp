#include "../../../example/fem/hex_hermite_mesh_energy.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace {
using namespace litepgo;

Vec affine_dofs(const Vec& X_nodes, const Eigen::Matrix3d& A, const Eigen::Vector3d& b = Eigen::Vector3d::Zero()) {
    Vec x = Vec::Zero(8 * X_nodes.size());
    for (int v = 0; v < X_nodes.size() / 3; ++v) {
        x.segment<3>(24 * v) = A * X_nodes.segment<3>(3 * v) + b;
        for (int d = 0; d < 3; ++d)
            x.segment<3>(24 * v + 3 * (d + 1)) = A.col(d);
    }
    return x;
}
} // namespace

TEST(HexHermiteExample, ReproducesAffineDisplacementAndVolumeWeightedEnergy) {
    const Vec X_nodes = example::hex_hermite_rest_positions();
    const auto topology = example::hex_hermite_topology();
    const VertexDofMap geometry_map(topology, 3), displacement_map(topology, 24);
    Eigen::Matrix3d A;
    A << 0.1, 0.02, -0.03, 0.04, -0.05, 0.02, 0.01, 0.03, 0.08;
    const Eigen::Vector3d b(0.2, -0.1, 0.3);
    const Vec x = affine_dofs(X_nodes, A, b);
    auto energy = example::make_hex_hermite_mesh_energy(X_nodes);
    EXPECT_DOUBLE_EQ(energy.value(Vec::Zero(x.size())), 0);
    EXPECT_NEAR(
        energy.value(x),
        18 * NeoHookean::value({Eigen::Matrix3d::Identity() + A, Eigen::Matrix3d::Identity(), Eigen::Vector2d(2, 3)}),
        1e-11);
    const std::array<Eigen::Vector3d, 3> points{
        {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1), Eigen::Vector3d(0.2, 0.3, 0.4)}};
    for (int e = 0; e < 2; ++e) {
        const Interpolation<HexQ1Basis> geometry(geometry_map.gather(e, X_nodes));
        const auto coefficients = example::hex_hermite_reference_coefficients(
            displacement_map.gather(e, x), hex_hermite_dof_transform(geometry).diagonal());
        const Interpolation<HexHermiteBasis> displacement(coefficients);
        const DeformationGradientField<HexHermiteBasis> F(coefficients);
        for (const auto& xi : points) {
            const auto point = rest_point(geometry, xi);
            EXPECT_TRUE(displacement.value(xi).isApprox(A * point.X + b, 1e-13));
            EXPECT_TRUE(F.value(point).isApprox(Eigen::Matrix3d::Identity() + A, 1e-13));
        }
    }
}

TEST(HexHermiteExample, SharedFacesHaveContinuousValuesAndPhysicalGradientsInAllDirections) {
    for (int normal = 0; normal < 3; ++normal) {
        SCOPED_TRACE(normal);
        std::vector<HexTopology::Cell> cells(2);
        std::vector<Eigen::Vector3d> vertices;
        for (int e = 0; e < 2; ++e) {
            Eigen::Vector3d lengths(2, 3, 4), origin = Eigen::Vector3d::Zero();
            lengths[normal] = e + 1;
            origin[normal] = e;
            for (int v = 0; v < 8; ++v) {
                const Eigen::Vector3d X = origin + lengths.cwiseProduct(HexQ1Basis::reference_node(v));
                const auto found = std::find(vertices.begin(), vertices.end(), X);
                cells[static_cast<std::size_t>(e)][static_cast<std::size_t>(v)] =
                    static_cast<int>(found - vertices.begin());
                if (found == vertices.end())
                    vertices.push_back(X);
            }
        }
        ASSERT_EQ(vertices.size(), 12);
        Vec X_nodes(36), x(288);
        for (int v = 0; v < 12; ++v)
            X_nodes.segment<3>(3 * v) = vertices[static_cast<std::size_t>(v)];
        for (int j = 0; j < x.size(); ++j)
            x[j] = 0.01 * std::sin(0.7 * j + 0.3);
        const HexTopology topology(12, cells);
        const VertexDofMap geometry_map(topology, 3), displacement_map(topology, 24);
        const std::array<Interpolation<HexQ1Basis>, 2> geometries{
            Interpolation<HexQ1Basis>(geometry_map.gather(0, X_nodes)),
            Interpolation<HexQ1Basis>(geometry_map.gather(1, X_nodes))};
        const auto c0 = example::hex_hermite_reference_coefficients(
            displacement_map.gather(0, x), hex_hermite_dof_transform(geometries[0]).diagonal());
        const auto c1 = example::hex_hermite_reference_coefficients(
            displacement_map.gather(1, x), hex_hermite_dof_transform(geometries[1]).diagonal());
        const Interpolation<HexHermiteBasis> u0(c0), u1(c1);
        const DeformationGradientField<HexHermiteBasis> F0(c0), F1(c1);
        for (double s : {0.0, 0.23, 0.67, 1.0})
            for (double t : {0.0, 0.31, 0.8, 1.0}) {
                Eigen::Vector3d xi = Eigen::Vector3d::Zero();
                xi[(normal + 1) % 3] = s;
                xi[(normal + 2) % 3] = t;
                xi[normal] = 1;
                const auto p0 = rest_point(geometries[0], xi);
                xi[normal] = 0;
                const auto p1 = rest_point(geometries[1], xi);
                EXPECT_TRUE(p0.X.isApprox(p1.X, 1e-14));
                EXPECT_LT((u0.value(p0.xi) - u1.value(p1.xi)).norm(), 1e-13);
                EXPECT_LT((F0.value(p0) - F1.value(p1)).norm(), 1e-13);
            }
    }
}

TEST(HexHermiteExample, PhysicalDofGradientAndHessianMatchFiniteDifferences) {
    const Vec X_nodes = example::hex_hermite_rest_positions();
    auto energy = example::make_hex_hermite_mesh_energy(X_nodes);
    Vec x = affine_dofs(X_nodes, 0.1 * Eigen::Matrix3d::Identity());
    for (int j = 0; j < x.size(); ++j)
        x[j] += 0.005 * std::sin(0.7 * j + 0.3);
    ASSERT_EQ(x.size(), 288);
    Vec g;
    SpMat sparse_h = energy.hessian_pattern();
    const double value = energy.evaluate(x, &g, &sparse_h);
    const Eigen::MatrixXd H(sparse_h);
    ASSERT_EQ(g.size(), x.size());
    ASSERT_EQ(H.rows(), x.size());
    ASSERT_EQ(H.cols(), x.size());
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    EXPECT_NEAR(energy.value(x), value, 1e-13);

    constexpr double h = 1e-4;
    Vec fd_g(x.size());
    Eigen::MatrixXd fd_h(x.size(), x.size());
    for (int j = 0; j < x.size(); ++j) {
        Vec plus = x, minus = x, plus_g, minus_g;
        plus[j] += h;
        minus[j] -= h;
        fd_g[j] = (energy.evaluate(plus, &plus_g, nullptr) - energy.evaluate(minus, &minus_g, nullptr)) / (2 * h);
        fd_h.col(j) = (plus_g - minus_g) / (2 * h);
    }
    EXPECT_LT((g - fd_g).norm() / (1 + g.norm()), 2e-7);
    EXPECT_LT((H - fd_h).norm() / (1 + H.norm()), 2e-7);

    Vec d(x.size()), v(x.size());
    for (int j = 0; j < x.size(); ++j) {
        d[j] = std::sin(j + 0.5);
        v[j] = std::cos(j + 0.2);
    }
    d.normalize();
    v.normalize();
    for (double step : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(step);
        const double mixed = (energy.value(x + step * d + step * v) - energy.value(x + step * d - step * v) -
                              energy.value(x - step * d + step * v) + energy.value(x - step * d - step * v)) /
                             (4 * step * step);
        EXPECT_NEAR(d.dot(H * v), mixed, (10 * step * step + 1e-7) * (1 + H.norm()));
    }
    SpMat h_only = energy.hessian_pattern();
    EXPECT_NEAR(energy.evaluate(x, nullptr, &h_only), value, 1e-13);
    EXPECT_LT((Eigen::MatrixXd(h_only) - H).norm(), 1e-13 * (1 + H.norm()));
}

TEST(HexHermiteExample, RejectsUnsupportedGeometryAndIncorrectDofShapes) {
    for (int which = 0; which < 6; ++which) {
        SCOPED_TRACE(which);
        Vec X_nodes = example::hex_hermite_rest_positions();
        for (int v = 0; v < 12; ++v) {
            if (which == 0)
                X_nodes[3 * v] += 0.1 * X_nodes[3 * v + 1];
            if (which == 2)
                X_nodes[3 * v + 2] = 0;
            if (which == 3 || which == 4)
                X_nodes[3 * v] *= -1;
            if (which == 4)
                X_nodes[3 * v + 1] *= -1;
        }
        if (which == 1)
            X_nodes[3 * 6] += 0.1;
        if (which == 5)
            X_nodes[0] = std::numeric_limits<double>::quiet_NaN();
        EXPECT_THROW(example::make_hex_hermite_mesh_energy(X_nodes), std::invalid_argument);
    }
    EXPECT_THROW(example::hex_hermite_reference_coefficients(Eigen::MatrixXd::Zero(8, 3), Vec::Ones(192)),
                 std::invalid_argument);
    EXPECT_THROW(example::hex_hermite_reference_coefficients(Eigen::MatrixXd::Zero(8, 24), Vec::Ones(191)),
                 std::invalid_argument);
}
