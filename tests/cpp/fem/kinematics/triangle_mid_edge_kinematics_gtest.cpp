#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/mesh/triangle_mid_edge_stencils.hpp"

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
using namespace litepgo;

Eigen::MatrixXd curved_stencil() {
    Eigen::Matrix<double, 6, 3> x;
    x << 0, 0, 0, 1.2, 0.1, 0.2, -0.1, 0.9, -0.1, 1.1, 1.3, 0.4, -1, 0.2, -0.3, 0.2, -1.1, 0.3;
    return x;
}

Eigen::Matrix<double, 6, 1> pack(const TriangleMidEdgeKinematics& k) {
    Eigen::Matrix<double, 6, 1> result;
    result << k.a(0, 0), k.a(0, 1), k.a(1, 1), k.b(0, 0), k.b(0, 1), k.b(1, 1);
    return result;
}

void check_derivatives(const Eigen::MatrixXd& x, const std::array<int, 3>& opposite) {
    Eigen::MatrixXd D;
    std::array<Eigen::MatrixXd, 6> H;
    const auto k = eval_triangle_mid_edge_kinematics(x, opposite, &D, &H);
    EXPECT_EQ(D.rows(), 6);
    ASSERT_EQ(D.cols(), x.size());
    const auto value_only = eval_triangle_mid_edge_kinematics(x, opposite);
    EXPECT_TRUE(pack(k).isApprox(pack(value_only), 1e-14));
    for (const auto& h : H) {
        EXPECT_EQ(h.rows(), x.size());
        EXPECT_EQ(h.cols(), x.size());
        EXPECT_LT((h - h.transpose()).norm(), 1e-12 * (1 + h.norm()));
    }
    constexpr double step = 2e-6;
    for (int j = 0; j < x.size(); ++j) {
        Eigen::MatrixXd plus = x, minus = x;
        plus(j / 3, j % 3) += step;
        minus(j / 3, j % 3) -= step;
        Eigen::MatrixXd D_plus, D_minus;
        const auto a = eval_triangle_mid_edge_kinematics(plus, opposite, &D_plus);
        const auto b = eval_triangle_mid_edge_kinematics(minus, opposite, &D_minus);
        const Eigen::Matrix<double, 6, 1> fd = (pack(a) - pack(b)) / (2 * step);
        EXPECT_LT((fd - D.col(j)).norm(), 2e-8);
        const Eigen::MatrixXd fd_H = (D_plus - D_minus) / (2 * step);
        for (int i = 0; i < 6; ++i)
            EXPECT_LT((fd_H.row(i).transpose() - H[static_cast<std::size_t>(i)].col(j)).norm(), 2e-7);
    }
    Eigen::MatrixXd D_only;
    const auto first = eval_triangle_mid_edge_kinematics(x, opposite, &D_only);
    std::array<Eigen::MatrixXd, 6> H_only;
    const auto second = eval_triangle_mid_edge_kinematics(x, opposite, nullptr, &H_only);
    EXPECT_EQ(D_only, D);
    EXPECT_EQ(first.a, k.a);
    EXPECT_EQ(second.b, k.b);
    for (std::size_t i = 0; i < H.size(); ++i)
        EXPECT_EQ(H_only[i], H[i]);
}

TEST(TriangleMidEdgeKinematics, FlatStencilHasExpectedMetricAndZeroCurvature) {
    Eigen::MatrixXd x = curved_stencil();
    x.col(2).setZero();
    const auto k = eval_triangle_mid_edge_kinematics(x, {3, 4, 5});
    Eigen::Matrix2d a;
    a << 1.45, -0.03, -0.03, 0.82;
    EXPECT_TRUE(k.a.isApprox(a, 1e-14));
    EXPECT_TRUE(k.b.isZero());
    const auto boundary = eval_triangle_mid_edge_kinematics(x.topRows(3), {-1, -1, -1});
    EXPECT_EQ(boundary.a, k.a);
    EXPECT_TRUE(boundary.b.isZero());
}

TEST(TriangleMidEdgeKinematics, FoldedHingeHasExpectedCurvatureAndSign) {
    Eigen::Matrix<double, 4, 3> x;
    x << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1;
    for (double height : {0.5, 1.0, 2.0}) {
        x(3, 2) = height;
        const auto k = eval_triangle_mid_edge_kinematics(x, {-1, -1, 3});
        EXPECT_EQ(k.a, Eigen::Matrix2d::Identity());
        Eigen::Matrix2d b = Eigen::Matrix2d::Zero();
        b(1, 1) = 2 * height / std::sqrt(1 + height * height);
        EXPECT_TRUE(k.b.isApprox(b, 1e-14));
    }
    x.row(3) << 1, 1, 1;
    const auto mixed = eval_triangle_mid_edge_kinematics(x, {3, -1, -1});
    EXPECT_TRUE(mixed.b.isApprox(Eigen::Matrix2d::Constant(2 / std::sqrt(6.0)), 1e-14));
}

TEST(TriangleMidEdgeKinematics, IsInvariantUnderRigidMotionsAndScalesGeometrically) {
    const Eigen::MatrixXd x = curved_stencil();
    const auto k = eval_triangle_mid_edge_kinematics(x, {3, 4, 5});
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.7, Eigen::Vector3d(1, 2, -1).normalized()).toRotationMatrix();
    const Eigen::MatrixXd moved = (x * R.transpose()).rowwise() + Eigen::RowVector3d(2, -1, 3);
    const auto rigid = eval_triangle_mid_edge_kinematics(moved, {3, 4, 5});
    EXPECT_TRUE(rigid.a.isApprox(k.a, 1e-13));
    EXPECT_TRUE(rigid.b.isApprox(k.b, 1e-13));
    const auto scaled = eval_triangle_mid_edge_kinematics(2.3 * x, {3, 4, 5});
    EXPECT_TRUE(scaled.a.isApprox(2.3 * 2.3 * k.a, 1e-13));
    EXPECT_TRUE(scaled.b.isApprox(2.3 * k.b, 1e-13));
}

TEST(TriangleMidEdgeKinematics, CyclicFaceOrderingTransformsBothFormsAsTensors) {
    const Eigen::MatrixXd x = curved_stencil();
    const auto k = eval_triangle_mid_edge_kinematics(x, {3, 4, 5});
    Eigen::MatrixXd reordered = x;
    reordered.topRows(3) << x.row(1), x.row(2), x.row(0);
    const auto other = eval_triangle_mid_edge_kinematics(reordered, {4, 5, 3});
    Eigen::Matrix2d T;
    T << -1, -1, 1, 0;
    EXPECT_TRUE(other.a.isApprox(T.transpose() * k.a * T, 1e-13));
    EXPECT_TRUE(other.b.isApprox(T.transpose() * k.b * T, 1e-13));
}

TEST(TriangleMidEdgeKinematics, FullAndBoundaryStencilDerivativesMatchFiniteDifferences) {
    const Eigen::MatrixXd x = curved_stencil();
    check_derivatives(x, {3, 4, 5});
    check_derivatives(x.topRows(4), {3, -1, -1});
    check_derivatives(x.topRows(3), {-1, -1, -1});
}

TEST(TriangleMidEdgeKinematics, EveryBoundaryMaskHasCorrectDerivativesAndUnusedRowsStayZero) {
    const Eigen::MatrixXd x = curved_stencil();
    for (int mask = 0; mask < 8; ++mask) {
        SCOPED_TRACE(mask);
        Eigen::MatrixXd compact = x;
        std::array<int, 3> opposite{-1, -1, -1};
        int rows = 3;
        for (int k = 0; k < 3; ++k)
            if (mask & (1 << k)) {
                opposite[k] = rows;
                compact.row(rows++) = x.row(3 + k);
            }
        compact.conservativeResize(rows, 3);
        check_derivatives(compact, opposite);
    }
    Eigen::MatrixXd D;
    std::array<Eigen::MatrixXd, 6> H;
    eval_triangle_mid_edge_kinematics(x, {3, 3, -1}, &D, &H);
    EXPECT_TRUE(D.rightCols(6).isZero(0));
    for (const auto& h : H) {
        EXPECT_TRUE(h.bottomRows(6).isZero(0));
        EXPECT_TRUE(h.rightCols(6).isZero(0));
    }
}

TEST(TriangleMidEdgeKinematics, DerivativesRespectTranslationAndHomogeneousScaling) {
    for (const auto& opposite : {std::array<int, 3>{3, 4, 5}, std::array<int, 3>{3, 3, -1}}) {
        const Eigen::MatrixXd x = curved_stencil();
        Eigen::MatrixXd D;
        std::array<Eigen::MatrixXd, 6> H;
        const auto ab = pack(eval_triangle_mid_edge_kinematics(x, opposite, &D, &H));
        const Eigen::VectorXd positions = x.reshaped<Eigen::RowMajor>();
        for (int i = 0; i < 6; ++i) {
            const double degree = i < 3 ? 2.0 : 1.0;
            EXPECT_NEAR(D.row(i).dot(positions), degree * ab[i], 1e-12);
            EXPECT_LT((H[i] * positions - (degree - 1) * D.row(i).transpose()).norm(), 1e-12);
        }
        for (int component = 0; component < 3; ++component) {
            Eigen::VectorXd translation = Eigen::VectorXd::Zero(x.size());
            for (int v = 0; v < x.rows(); ++v)
                translation[3 * v + component] = 1;
            EXPECT_TRUE((D * translation).isZero(1e-12));
            for (const auto& h : H)
                EXPECT_TRUE((h * translation).isZero(1e-12));
        }
    }
}

TEST(TriangleMidEdgeKinematics, DeduplicatedClosedStencilDerivativesMatchFiniteDifferences) {
    const TriangleTopology topology(4, {{0, 1, 2}, {0, 3, 1}, {1, 3, 2}, {2, 3, 0}});
    const auto stencils = make_triangle_mid_edge_stencils(topology);
    const StencilDofMap map(topology.num_vertices(), stencils.vertices, 3);
    Eigen::Matrix<double, 4, 3> X;
    X << 0, 0, 0, 1.2, 0, 0, 0, 0.9, 0, 0.1, 0.2, 1.1;
    const Vec packed = X.reshaped<Eigen::RowMajor>();
    for (int f = 0; f < map.num_elements(); ++f)
        check_derivatives(map.gather(f, packed), stencils.opposite[static_cast<std::size_t>(f)]);
}

TEST(TriangleMidEdgeKinematics, RejectsInvalidLayoutsAndNonFiniteInputs) {
    const Eigen::MatrixXd x = curved_stencil();
    EXPECT_THROW(eval_triangle_mid_edge_kinematics(Eigen::MatrixXd::Zero(2, 3), {-1, -1, -1}), std::invalid_argument);
    EXPECT_THROW(eval_triangle_mid_edge_kinematics(Eigen::MatrixXd::Zero(7, 3), {3, 4, 5}), std::invalid_argument);
    EXPECT_THROW(eval_triangle_mid_edge_kinematics(Eigen::MatrixXd::Zero(3, 2), {-1, -1, -1}), std::invalid_argument);
    for (int p : {-2, 0, 2, 6})
        EXPECT_THROW(eval_triangle_mid_edge_kinematics(x, {p, 4, 5}), std::invalid_argument);
    for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        Eigen::MatrixXd bad = x;
        bad(0, 0) = value;
        EXPECT_THROW(eval_triangle_mid_edge_kinematics(bad, {3, 4, 5}), std::invalid_argument);
    }
}

TEST(TriangleMidEdgeKinematics, RejectsDegenerateFacesCancellingNormalsAndOverflow) {
    const Eigen::MatrixXd x = curved_stencil();
    Eigen::MatrixXd bad = x;
    bad.row(2) = bad.row(0);
    EXPECT_THROW(eval_triangle_mid_edge_kinematics(bad, {3, 4, 5}), InfeasibleState);
    bad = x;
    bad.row(3) = bad.row(1);
    EXPECT_THROW(eval_triangle_mid_edge_kinematics(bad, {3, 4, 5}), InfeasibleState);
    Eigen::Matrix<double, 4, 3> cancelling;
    cancelling << 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0;
    EXPECT_THROW(eval_triangle_mid_edge_kinematics(cancelling, {-1, -1, 3}), InfeasibleState);
    EXPECT_THROW(eval_triangle_mid_edge_kinematics(1e200 * x, {3, 4, 5}), InfeasibleState);
}
} // namespace
