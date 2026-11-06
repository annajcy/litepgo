#include "litepgo/fem/kinematics/volumetric_element.hpp"
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

struct QuadraticLocalEnergy : LocalEnergy3D {
    Eigen::VectorXd coefficients = Eigen::Vector2d(0.2, -0.3);
    int n = 2;
    int num_dofs() const override {
        return n;
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }
    double eval_value(const Eigen::Vector3d& xi) const override {
        return 0.5 * (1 + xi.x()) * coefficients.squaredNorm();
    }
    Eigen::VectorXd eval_gradient(const Eigen::Vector3d& xi) const override {
        return (1 + xi.x()) * coefficients;
    }
    Eigen::MatrixXd eval_hessian(const Eigen::Vector3d& xi) const override {
        return (1 + xi.x()) * Eigen::MatrixXd::Identity(n, n);
    }
    double eval_feasible_step(const Eigen::Vector3d& xi, const Eigen::Ref<const Eigen::VectorXd>&) const {
        return 0.2 + 0.3 * xi.z();
    }
};

TEST(VolumetricElement3D, IntegratesNonMaterialLocalEnergyWithItsOwnDofCountAndStep) {
    ReferenceResponse X;
    X.J_X = Eigen::Vector3d(2, 3, 4).asDiagonal();
    const VolumetricElement3D<GaussLegendre3D<2>> element(X);
    const QuadraticLocalEnergy local;
    const LocalEnergy3D& base = local;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    EXPECT_NEAR(element.eval(base, &g, &H), 18 * local.coefficients.squaredNorm(), 1e-14);
    EXPECT_TRUE(g.isApprox(36 * local.coefficients, 1e-14));
    EXPECT_TRUE(H.isApprox(36 * Eigen::Matrix2d::Identity(), 1e-14));
    EXPECT_DOUBLE_EQ(element.max_feasible_step(local, Eigen::Vector2d::Ones()),
                     0.2 + 0.3 * GaussLegendre3D<2>::points().col(2).minCoeff());
    EXPECT_THROW(element.max_feasible_step(local, Eigen::Vector3d::Zero()), std::invalid_argument);
    EXPECT_THROW(element.max_feasible_step(local, Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN())),
                 std::invalid_argument);
    QuadraticLocalEnergy empty;
    empty.n = 0;
    EXPECT_THROW(element.eval(empty), std::invalid_argument);
}

TEST(VolumetricElement3D, OwnsIntegrationWeightsWithoutRetainingReferenceResponse) {
    ReferenceResponse X;
    const VolumetricElement3D<GaussLegendre3D<2>> element(X);
    EXPECT_EQ(X.queries, 8);
    QuadraticLocalEnergy local;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    const double E = element.eval(local, &g, &H);
    X.J_X *= 2;
    EXPECT_DOUBLE_EQ(element.eval(local), E);
    EXPECT_EQ(X.queries, 8);
    const VolumetricElement3D<GaussLegendre3D<2>> rebuilt(X);
    EXPECT_DOUBLE_EQ(rebuilt.eval(local), 8 * E);
    const auto detached = [] {
        ReferenceResponse X;
        return VolumetricElement3D<GaussLegendre3D<2>>(X);
    }();
    const auto copied = detached;
    tbb::parallel_for(0, 8, [&](int) {
        Eigen::VectorXd local_g;
        Eigen::MatrixXd local_H;
        EXPECT_DOUBLE_EQ(copied.eval(local, &local_g, &local_H), E);
        EXPECT_EQ(local_g, g);
        EXPECT_EQ(local_H, H);
    });
}

struct ValueOnlyLocalEnergy : QuadraticLocalEnergy {
    bool has_gradient() const noexcept override {
        return false;
    }
    bool has_hessian() const noexcept override {
        return false;
    }
};
TEST(VolumetricElement3D, ChecksLocalCapabilitiesBeforeWritingOutputs) {
    ReferenceResponse X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    const ValueOnlyLocalEnergy local;
    EXPECT_TRUE(std::isfinite(element.eval(local)));
    Eigen::VectorXd g = Eigen::VectorXd::Constant(1, 7);
    Eigen::MatrixXd H = Eigen::MatrixXd::Constant(1, 2, 9);
    EXPECT_THROW(element.eval(local, &g), std::logic_error);
    EXPECT_THROW(element.eval(local, nullptr, &H), std::logic_error);
    EXPECT_EQ(g, Eigen::VectorXd::Constant(1, 7));
    EXPECT_EQ(H, Eigen::MatrixXd::Constant(1, 2, 9));
}

struct InvalidAccumulation : QuadraticLocalEnergy {
    int mode = 0;
    double accumulate_into(const Eigen::Vector3d&, double, StridedVecRef* g, StridedMatRef* H) const override {
        const double infinity = std::numeric_limits<double>::infinity();
        if (g)
            g->setConstant(mode == 3 ? infinity : 0);
        if (H)
            H->setConstant(mode == 4 ? infinity : 0);
        return mode == 0 ? std::numeric_limits<double>::max() : 0;
    }
};
TEST(VolumetricElement3D, RejectsMalformedAndNonFiniteLocalAccumulationResults) {
    ReferenceResponse X;
    const VolumetricElement3D<GaussLegendre3D<2>> element(X);
    InvalidAccumulation local;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    EXPECT_THROW(element.eval(local), std::domain_error);
    for (int mode : {3, 4}) {
        local.mode = mode;
        EXPECT_THROW(element.eval(local, &g, &H), std::domain_error);
    }
}

struct InvalidStepLocalEnergy : QuadraticLocalEnergy {
    double step = 0;
    double eval_feasible_step(const Eigen::Vector3d&, const Eigen::Ref<const Eigen::VectorXd>&) const {
        return step;
    }
};
TEST(VolumetricElement3D, RejectsInvalidLocalStepResults) {
    ReferenceResponse X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    InvalidStepLocalEnergy local;
    for (double bad : {-1., 2., std::numeric_limits<double>::quiet_NaN()}) {
        local.step = bad;
        EXPECT_THROW(element.max_feasible_step(local, Eigen::Vector2d::Zero()), std::logic_error);
    }
}

} // namespace

} // namespace litepgo::fem::test

TEST(VolumetricElementViews, IntegratesIntoPaddedStridedOutputsWithoutTouchingNeighbors) {
    using namespace litepgo;
    using namespace litepgo::fem;
    Eigen::Matrix<double, 4, 3> nodes;
    for (int i = 0; i < 4; ++i)
        nodes.row(i) = TetP1Basis::coefficient_info(i).point.transpose();
    InterpolatedResponse<TetP1Basis> X(nodes);
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    litepgo::fem::test::QuadraticLocalEnergy local;
    local.n = 9;
    local.coefficients = Vec::LinSpaced(9, 0.1, 0.9);
    Vec expected_g;
    Eigen::MatrixXd expected_H;
    const double expected = element.eval(local, &expected_g, &expected_H);
    Vec owned_g = Vec::Constant(1, -123);
    Eigen::MatrixXd owned_H = Eigen::MatrixXd::Constant(2, 3, -123);
    for (int mask = 0; mask < 4; ++mask) {
        EXPECT_DOUBLE_EQ(element.eval(local, mask & 1 ? &owned_g : nullptr, mask & 2 ? &owned_H : nullptr), expected);
        if (mask & 1)
            EXPECT_TRUE(owned_g.isApprox(expected_g));
        if (mask & 2)
            EXPECT_TRUE(owned_H.isApprox(expected_H));
    }
    Vec storage = Vec::Constant(22, -123);
    Eigen::Map<Vec, 0, Eigen::InnerStride<2>> mapped_g(storage.data() + 2, 9);
    StridedVecRef g(mapped_g);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix =
        Eigen::MatrixXd::Constant(11, 13, -123);
    Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> mapped_H(
        matrix.data() + 15, 9, 9, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, 13));
    StridedMatRef H(mapped_H);
    EXPECT_DOUBLE_EQ(element.eval_into(local, &g, &H), expected);
    EXPECT_TRUE(g.isApprox(expected_g));
    EXPECT_TRUE(H.isApprox(expected_H));
    for (int i = 0; i < storage.size(); ++i)
        if (i < 2 || i > 18 || i % 2 != 0)
            EXPECT_DOUBLE_EQ(storage[i], -123);
    for (int i = 0; i < matrix.rows(); ++i)
        for (int j = 0; j < matrix.cols(); ++j)
            if (i < 1 || i > 9 || j < 2 || j > 10)
                EXPECT_DOUBLE_EQ(matrix(i, j), -123);
    EXPECT_DOUBLE_EQ(element.eval_into(local, nullptr, &H), expected);
    EXPECT_DOUBLE_EQ(element.eval_into(local, &g, nullptr), expected);
    EXPECT_DOUBLE_EQ(element.eval_into(local, nullptr, nullptr), expected);
    EXPECT_TRUE(g.isApprox(expected_g));
    EXPECT_TRUE(H.isApprox(expected_H));
    StridedVecRef wrong(storage.head(8));
    EXPECT_THROW(element.eval_into(local, &wrong, &H), std::invalid_argument);
    EXPECT_TRUE(H.isApprox(expected_H));
}

TEST(VolumetricElementViews, ResponseWritesPaddedRectangularBatchBlocks) {
    using namespace litepgo;
    using namespace litepgo::fem;
    Eigen::Matrix<double, 4, 3> nodes;
    for (int i = 0; i < 4; ++i)
        nodes.row(i) = TetP1Basis::coefficient_info(i).point.transpose();
    InterpolatedResponse<TetP1Basis> X(nodes);
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    struct LinearLocal : LocalResponse3D {
        int num_inputs() const override {
            return 2;
        }
        int num_outputs() const override {
            return 9;
        }
        bool has_jacobian() const noexcept override {
            return true;
        }
        Vec eval_value(const Eigen::Vector3d& xi) const override {
            return Vec::Constant(9, xi.sum());
        }
        Eigen::MatrixXd eval_jacobian(const Eigen::Vector3d&) const override {
            return Eigen::MatrixXd::Ones(9, 2);
        }
    } local;
    Eigen::MatrixXd expected_J;
    const Vec expected = element.eval(local, &expected_J);
    Eigen::MatrixXd owned_J = Eigen::MatrixXd::Constant(2, 3, -123);
    EXPECT_TRUE(element.eval(local, &owned_J).isApprox(expected));
    EXPECT_TRUE(owned_J.isApprox(expected_J));
    EXPECT_TRUE(element.eval(local).isApprox(expected));
    Eigen::MatrixXd values = Eigen::MatrixXd::Constant(2, 9, -123);
    StridedVecRef v(values.row(1).transpose());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> storage =
        Eigen::MatrixXd::Constant(11, 4, -123);
    Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> block(
        storage.data() + 5, 9, 2, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, 4));
    StridedMatRef J(block);
    element.eval_into(local, v, &J);
    EXPECT_TRUE(v.isApprox(expected));
    EXPECT_TRUE(J.isApprox(expected_J));
    EXPECT_TRUE(values.row(0).isConstant(-123));
    EXPECT_TRUE(storage.row(0).isConstant(-123));
    EXPECT_TRUE(storage.row(10).isConstant(-123));
    EXPECT_TRUE(storage.col(0).isConstant(-123));
    EXPECT_TRUE(storage.col(3).isConstant(-123));
    element.eval_into(local, v);
    EXPECT_TRUE(v.isApprox(expected));
    StridedVecRef wrong(values.row(0).head(8).transpose());
    EXPECT_THROW(element.eval_into(local, wrong, &J), std::invalid_argument);
    EXPECT_TRUE(J.isApprox(expected_J));
}

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
