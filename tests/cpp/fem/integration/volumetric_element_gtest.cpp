#include "litepgo/fem/integration/volumetric_element.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"

#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <tbb/parallel_for.h>

#include <cmath>
#include <limits>
#include <utility>

namespace litepgo::fem::test {

namespace {

struct ReferenceResponse : energy::Response {
    int components = 3, queries = 0;
    Eigen::MatrixXd J_X = Eigen::Matrix3d::Identity();
    int num_inputs() const noexcept override {
        return 3;
    }
    int num_outputs() const noexcept override {
        return components;
    }
    Vec eval_value_impl(const ConstVecRef&) override {
        return Vec::Zero(components);
    }
    std::pair<Vec, SpMat> eval_value_and_jacobian_impl(const ConstVecRef& x) override {
        ++queries;
        return {eval_value_impl(x), J_X.sparseView()};
    }
};

struct TestQuadrature {
    static constexpr int num_points = 1;
    inline static Eigen::Matrix<double, 1, 3> point = Eigen::Matrix<double, 1, 3>::Zero();
    inline static Eigen::Matrix<double, 1, 1> weight = Eigen::Matrix<double, 1, 1>::Ones();
    static const Eigen::Matrix<double, 1, 3>& points() {
        return point;
    }
    static const Eigen::Matrix<double, 1, 1>& weights() {
        return weight;
    }
};

TEST(VolumetricElement3D, RejectsInvalidReferenceGeometryAndQuadratureAtConstruction) {
    using Element3D = VolumetricElement3D<TetCentroidQuadrature>;
    ReferenceResponse X;
    X.components = 2;
    EXPECT_ANY_THROW(Element3D{X});
    X.components = 3;
    X.J_X.resize(2, 3);
    EXPECT_ANY_THROW(Element3D{X});
    for (double scale :
         {0., -1., 1e200, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        X.J_X = scale * Eigen::Matrix3d::Identity();
        EXPECT_ANY_THROW(Element3D{X});
    }
    X.J_X = Eigen::Vector3d(1e-310, 1, 1).asDiagonal();
    EXPECT_NO_THROW(Element3D{X});
    X.J_X = 1e-100 * Eigen::Matrix3d::Identity();
    EXPECT_NO_THROW(Element3D{X});
    X.J_X = Eigen::Matrix3d::Identity();
    using QuadratureElement3D = VolumetricElement3D<TestQuadrature>;
    TestQuadrature::point[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(QuadratureElement3D{X}, std::invalid_argument);
    TestQuadrature::point.setZero();
    TestQuadrature::weight[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(QuadratureElement3D{X}, std::invalid_argument);
    TestQuadrature::weight[0] = std::numeric_limits<double>::max();
    X.J_X = 2 * Eigen::Matrix3d::Identity();
    EXPECT_THROW(QuadratureElement3D{X}, std::invalid_argument);
    TestQuadrature::weight.setOnes();
}

TEST(VolumetricElement3D, OwnsWeightsWithoutRetainingGeometry) {
    ReferenceResponse X;
    const VolumetricElement3D<GaussLegendre3D<2>> element(X);
    EXPECT_EQ(X.queries, 8);
    const double weight = element.point(0).w_dV;
    X.J_X *= 2;
    EXPECT_DOUBLE_EQ(element.point(0).w_dV, weight);
    const VolumetricElement3D<GaussLegendre3D<2>> rebuilt(X);
    EXPECT_DOUBLE_EQ(rebuilt.point(0).w_dV, 8 * weight);
    const auto detached = [] {
        ReferenceResponse X;
        return VolumetricElement3D<GaussLegendre3D<2>>(X);
    }();
    const auto copied = detached;
    tbb::parallel_for(0, 8, [&](int q) { EXPECT_DOUBLE_EQ(copied.point(q).w_dV, element.point(q).w_dV); });
    EXPECT_THROW(element.point(-1), std::out_of_range);
    EXPECT_THROW(element.point(8), std::out_of_range);
}

TEST(VolumetricElement3D, CallbackWritesStridedViewsWithoutTouchingNeighbors) {
    ReferenceResponse X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    Vec storage = Vec::Constant(8, -123);
    Eigen::Map<Vec, 0, Eigen::InnerStride<2>> mapped(storage.data() + 1, 3);
    StridedVecRef g(mapped);
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Constant(5, 5, -123);
    StridedMatRef H(matrix.block(1, 1, 3, 3));
    auto accumulate = [](int, const auto& p, StridedVecRef* dg, StridedMatRef* dH) {
        if (dg)
            dg->array() += 2 * p.w_dV;
        if (dH)
            dH->diagonal().array() += 3 * p.w_dV;
        return p.w_dV;
    };
    for (int mask = 0; mask < 4; ++mask) {
        EXPECT_DOUBLE_EQ(element.eval_into(3, accumulate, mask & 1 ? &g : nullptr, mask & 2 ? &H : nullptr), 1. / 6);
        if (mask & 1)
            EXPECT_TRUE(g.isConstant(1. / 3));
        if (mask & 2)
            EXPECT_TRUE(H.isApprox(0.5 * Eigen::Matrix3d::Identity()));
    }
    for (int i = 0; i < 8; ++i)
        if (i % 2 == 0 || i == 7)
            EXPECT_EQ(storage[i], -123);
    EXPECT_TRUE(matrix.row(0).isConstant(-123));
    EXPECT_TRUE(matrix.row(4).isConstant(-123));
    EXPECT_TRUE(matrix.col(0).isConstant(-123));
    EXPECT_TRUE(matrix.col(4).isConstant(-123));
}
} // namespace
} // namespace litepgo::fem::test

TEST(VolumetricElementViews, IndexedCallbacksShareWeightsZeroingAndValidation) {
    using namespace litepgo;
    using namespace litepgo::fem;
    Eigen::Matrix<double, 8, 3> nodes;
    for (int i = 0; i < 8; ++i)
        nodes.row(i) = HexQ1Basis::coefficient_info(i).point.transpose();
    InterpolatedResponse<HexQ1Basis> X(nodes);
    const VolumetricElement3D<GaussLegendre3D<2>> element(X);
    Eigen::Vector2d gradient = Eigen::Vector2d::Constant(123);
    Eigen::Matrix2d hessian = Eigen::Matrix2d::Constant(123);
    StridedVecRef g(gradient);
    StridedMatRef H(hessian);
    int visited = 0;
    auto accumulate = [&](int q, const auto& point, StridedVecRef* dg, StridedMatRef* dH) {
        EXPECT_EQ(q, visited++);
        EXPECT_TRUE(point.xi.isApprox(GaussLegendre3D<2>::points().row(q).transpose()));
        EXPECT_NEAR(point.w_dV, GaussLegendre3D<2>::weights()[q], 1e-14);
        if (dg)
            *dg += point.w_dV * point.xi.head(2);
        if (dH)
            *dH += 2 * point.w_dV * Eigen::Matrix2d::Identity();
        return point.w_dV * (q + 1);
    };
    for (int mask = 0; mask < 4; ++mask) {
        visited = 0;
        EXPECT_NEAR(element.eval_into(2, accumulate, mask & 1 ? &g : nullptr, mask & 2 ? &H : nullptr), 4.5, 1e-14);
        EXPECT_EQ(visited, 8);
        if (mask & 1)
            EXPECT_TRUE(gradient.isApprox(Eigen::Vector2d::Constant(0.5), 1e-14));
        if (mask & 2)
            EXPECT_TRUE(hessian.isApprox(2 * Eigen::Matrix2d::Identity(), 1e-14));
    }
    visited = 0;
    EXPECT_THROW(element.eval_into(3, accumulate, &g, &H), std::invalid_argument);
    EXPECT_EQ(visited, 0);
    auto nonfinite = [](int, const auto&, StridedVecRef*, StridedMatRef*) {
        return std::numeric_limits<double>::quiet_NaN();
    };
    EXPECT_THROW(element.eval_into(2, nonfinite, nullptr, nullptr), std::domain_error);
    auto bad_gradient = [](int, const auto&, StridedVecRef* dg, StridedMatRef*) {
        (*dg)[0] = std::numeric_limits<double>::infinity();
        return 0.;
    };
    EXPECT_THROW(element.eval_into(2, bad_gradient, &g, nullptr), std::domain_error);
    visited = 0;
    EXPECT_NEAR(element.max_feasible_step([&](int q, const auto&) {
        EXPECT_EQ(q, visited++);
        return 0.95 - 0.1 * q;
    }),
                0.25, 1e-14);
    EXPECT_EQ(visited, 8);
    for (double bad : {-0.1, 1.1, std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(element.max_feasible_step([&](int, const auto&) { return bad; }), std::logic_error);
    }
}
