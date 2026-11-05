#include "litepgo/fem/kinematics/volumetric_element.hpp"

#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
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

template <class B, class Q>
double eval_displacement(const VolumetricElement3D<Q>& element, const Field3D& X, const Eigen::MatrixXd& u,
                         const Field3D& Fp, const Field3D& theta, Eigen::VectorXd* g = nullptr,
                         Eigen::MatrixXd* H = nullptr) {
    const InterpolatedField3D<B> displacement(u);
    const DeformationGradientField3D deformation(displacement, X);
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> local(deformation, Fp, theta);
    return element.eval(local, g, H);
}

template <class B, class Q>
double step_displacement(const VolumetricElement3D<Q>& element, const Field3D& X, const Eigen::MatrixXd& u,
                         const Eigen::MatrixXd& du, const Field3D& Fp, const Field3D& theta) {
    const InterpolatedField3D<B> displacement(u);
    const DeformationGradientField3D deformation(displacement, X);
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> local(deformation, Fp, theta);
    const Eigen::VectorXd delta = du.reshaped<Eigen::RowMajor>();
    return element.max_feasible_step(local, delta);
}

template <class B> Eigen::MatrixXd coordinates() {
    Eigen::MatrixXd X(B::num_basis_functions, 3);
    for (int a = 0; a < X.rows(); ++a) {
        const auto xi = B::coefficient_info(a).point;
        X.row(a) =
            Eigen::Vector3d(1.2 * xi.x() + 0.15 * xi.y() * xi.z(), 0.9 * xi.y() + 0.1 * xi.x() * xi.z(), 1.1 * xi.z())
                .transpose();
    }
    return X;
}

template <class B, class XBasis, class Q> void compare_uncached() {
    const InterpolatedField3D<XBasis> X(coordinates<XBasis>());
    const VolumetricElement3D<Q> element(X);
    Eigen::MatrixXd u(B::num_basis_functions, 3), du(u.rows(), 3);
    for (int a = 0; a < u.rows(); ++a)
        for (int c = 0; c < 3; ++c) {
            u(a, c) = 0.02 * std::sin(3 * a + c + 0.2);
            du(a, c) = 0.8 * std::cos(3 * a + c + 0.4);
        }
    const InterpolatedField3D<B> displacement(u);
    const DeformationGradientField3D deformation(displacement, X);
    const Eigen::Matrix3d Fp_matrix = (Eigen::Matrix3d() << 1.05, 0.02, 0, 0, 0.97, 0, 0, 0.01, 1.02).finished();
    const IdentityField3D Fp(Fp_matrix.reshaped());
    Eigen::MatrixXd theta_coeffs(XBasis::num_basis_functions, 2);
    for (int a = 0; a < theta_coeffs.rows(); ++a)
        theta_coeffs.row(a) = Eigen::Vector2d(2 + 0.2 * a, 3 + 0.1 * a).transpose();
    const InterpolatedField3D<XBasis> theta(theta_coeffs);
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> local(deformation, Fp, theta);
    Eigen::VectorXd g_ref = Eigen::VectorXd::Zero(local.num_dofs());
    Eigen::MatrixXd H_ref = Eigen::MatrixXd::Zero(local.num_dofs(), local.num_dofs());
    double E_ref = 0, alpha_ref = 1;
    const Eigen::VectorXd delta = du.reshaped<Eigen::RowMajor>();
    for (int q = 0; q < Q::num_points; ++q) {
        const Eigen::Vector3d xi = Q::points().row(q).transpose();
        const double weight = Q::weights()[q] * Eigen::Matrix3d(X.eval_dvalue_dxi(xi)).determinant();
        E_ref += local.accumulate(xi, weight, &g_ref, &H_ref);
        alpha_ref = std::min(alpha_ref, local.eval_feasible_step(xi, delta));
    }
    Eigen::VectorXd g = Eigen::VectorXd::Ones(1);
    Eigen::MatrixXd H = Eigen::MatrixXd::Ones(1, 2);
    const double E = eval_displacement<B>(element, X, u, Fp, theta, &g, &H);
    EXPECT_EQ(local.num_dofs(), 3 * B::num_basis_functions);
    EXPECT_NEAR(E, E_ref, 1e-12 * (1 + std::abs(E_ref)));
    EXPECT_LT((g - g_ref).norm(), 1e-12 * (1 + g_ref.norm()));
    EXPECT_LT((H - H_ref).norm(), 1e-12 * (1 + H_ref.norm()));
    EXPECT_NEAR(step_displacement<B>(element, X, u, du, Fp, theta), alpha_ref, 1e-9);
    EXPECT_DOUBLE_EQ(eval_displacement<B>(element, X, u, Fp, theta), E);
    Eigen::VectorXd g_only;
    Eigen::MatrixXd H_only;
    EXPECT_DOUBLE_EQ(eval_displacement<B>(element, X, u, Fp, theta, &g_only), E);
    EXPECT_DOUBLE_EQ(eval_displacement<B>(element, X, u, Fp, theta, nullptr, &H_only), E);
    EXPECT_EQ(g_only, g);
    EXPECT_EQ(H_only, H);
    EXPECT_DOUBLE_EQ(eval_displacement<B>(element, X, u, Fp, theta, &g, &H), E);
    EXPECT_EQ(g, g_only);
    EXPECT_EQ(H, H_only);

    constexpr double h = 1e-5;
    Eigen::VectorXd g_plus, g_minus;
    const double E_plus = eval_displacement<B>(element, X, u + h * du, Fp, theta, &g_plus);
    const double E_minus = eval_displacement<B>(element, X, u - h * du, Fp, theta, &g_minus);
    EXPECT_NEAR(g.dot(delta), (E_plus - E_minus) / (2 * h), 1e-7 * (1 + g.norm()));
    EXPECT_LT((H * delta - (g_plus - g_minus) / (2 * h)).norm(), 1e-7 * (1 + H.norm()));
}

TEST(VolumetricElement3D, TetMatchesUncachedFieldsAndDirectionalFiniteDifferences) {
    compare_uncached<TetP1Basis, TetP1Basis, TetCentroidQuadrature>();
}
TEST(VolumetricElement3D, WarpedHexMatchesUncachedFieldsAndDirectionalFiniteDifferences) {
    compare_uncached<HexQ1Basis, HexQ1Basis, GaussLegendre3D<2>>();
}
TEST(VolumetricElement3D, HermiteWithQ1GeometryMatchesUncachedFieldsAndDirectionalFiniteDifferences) {
    compare_uncached<HexHermiteBasis, HexQ1Basis, GaussLegendre3D<4>>();
}

struct ReferenceField : Field3D {
    int components = 3;
    Eigen::MatrixXd J_X = Eigen::Matrix3d::Identity();
    mutable int queries = 0;
    int num_components() const override {
        return components;
    }
    int num_dofs() const override {
        return 0;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        throw std::logic_error("not needed");
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        throw std::logic_error("not needed");
    }
    SpatialDerivative eval_dvalue_dxi(const Eigen::Vector3d&) const override {
        ++queries;
        return J_X;
    }
};

struct ValueField : Field3D {
    Eigen::VectorXd data;
    mutable int queries = 0;
    explicit ValueField(Eigen::VectorXd value) : data(std::move(value)) {}
    int num_components() const override {
        return static_cast<int>(data.size());
    }
    int num_dofs() const override {
        return 0;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        ++queries;
        return data;
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        throw std::logic_error("not needed");
    }
};

TEST(VolumetricElement3D, MaterialFieldsRemainLiveAndNeedOnlyValues) {
    const ReferenceField X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    Eigen::MatrixXd u(4, 3);
    for (int a = 0; a < 4; ++a)
        u.row(a) = 0.1 * TetP1Basis::coefficient_info(a).point.transpose();
    ValueField Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
    Eigen::VectorXd g, g_scaled;
    Eigen::MatrixXd H, H_scaled;
    const double E = eval_displacement<TetP1Basis>(element, X, u, Fp, theta, &g, &H);
    EXPECT_EQ(Fp.queries, 1);
    EXPECT_EQ(theta.queries, 1);
    theta.data *= 2;
    EXPECT_DOUBLE_EQ(eval_displacement<TetP1Basis>(element, X, u, Fp, theta, &g_scaled, &H_scaled), 2 * E);
    EXPECT_TRUE(g_scaled.isApprox(2 * g, 1e-14));
    EXPECT_TRUE(H_scaled.isApprox(2 * H, 1e-14));
    Fp.data = (1.1 * Eigen::Matrix3d::Identity()).reshaped();
    EXPECT_NEAR(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), 0, 1e-14);
    EXPECT_EQ(step_displacement<TetP1Basis>(element, X, u, Eigen::MatrixXd::Zero(4, 3), Fp, theta), 1);
    EXPECT_EQ(Fp.queries, 4);
    EXPECT_EQ(theta.queries, 4);
}

struct CountingMaterial {
    inline static int values = 0, gradients = 0, hessians = 0;
    static double eval_W(const VolumetricMaterialInputs& input) {
        ++values;
        return NeoHookean::eval_W(input);
    }
    static Eigen::Matrix3d eval_dW_dF(const VolumetricMaterialInputs& input) {
        ++gradients;
        return NeoHookean::eval_dW_dF(input);
    }
    static Eigen::Matrix<double, 9, 9> eval_d2W_dF2(const VolumetricMaterialInputs& input) {
        ++hessians;
        return NeoHookean::eval_d2W_dF2(input);
    }
};

TEST(VolumetricElement3D, EvaluatesOnlyRequestedLocalDerivativesAndDoesNotRequireAStepAPI) {
    const ReferenceField X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
    const InterpolatedField3D<TetP1Basis> displacement(Eigen::MatrixXd::Zero(4, 3));
    const DeformationGradientField3D deformation(displacement, X);
    const VolumetricMaterialLocalEnergyF3D<CountingMaterial> local(deformation, Fp, theta);
    CountingMaterial::values = CountingMaterial::gradients = CountingMaterial::hessians = 0;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    element.eval(local);
    EXPECT_EQ(CountingMaterial::gradients, 0);
    EXPECT_EQ(CountingMaterial::hessians, 0);
    element.eval(local, &g);
    EXPECT_EQ(CountingMaterial::gradients, 1);
    EXPECT_EQ(CountingMaterial::hessians, 0);
    element.eval(local, nullptr, &H);
    EXPECT_EQ(CountingMaterial::gradients, 1);
    EXPECT_EQ(CountingMaterial::hessians, 1);
    EXPECT_EQ(CountingMaterial::values, 3);
}

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
    ReferenceField X;
    X.components = 2;
    EXPECT_THROW(Element3D{X}, std::invalid_argument);
    X.components = 3;
    X.J_X.resize(2, 3);
    EXPECT_THROW(Element3D{X}, std::invalid_argument);
    for (double scale :
         {0., -1., 1e200, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        X.J_X = scale * Eigen::Matrix3d::Identity();
        EXPECT_THROW(Element3D{X}, std::invalid_argument);
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

TEST(VolumetricElement3D, RejectsMalformedCoefficientsAndMaterialFields) {
    const ReferenceField X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    const Eigen::MatrixXd u = Eigen::MatrixXd::Zero(4, 3);
    ValueField Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
    for (const Eigen::MatrixXd& bad :
         {Eigen::MatrixXd::Zero(3, 3).eval(), Eigen::MatrixXd::Zero(4, 2).eval(),
          Eigen::MatrixXd::Constant(4, 3, std::numeric_limits<double>::quiet_NaN()).eval()}) {
        EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, bad, Fp, theta), std::invalid_argument);
        EXPECT_THROW(step_displacement<TetP1Basis>(element, X, u, bad, Fp, theta), std::invalid_argument);
        EXPECT_THROW(step_displacement<TetP1Basis>(element, X, bad, u, Fp, theta), std::invalid_argument);
    }
    Fp.data.resize(8);
    EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), std::invalid_argument);
    Fp.data = Eigen::VectorXd::Constant(9, std::numeric_limits<double>::infinity());
    EXPECT_THROW(step_displacement<TetP1Basis>(element, X, u, u, Fp, theta), std::invalid_argument);
    Fp.data = Eigen::Matrix3d::Identity().reshaped();
    theta.data.resize(0);
    EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), std::invalid_argument);
    theta.data = Eigen::Vector2d(2, std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), std::invalid_argument);
    theta.data = Eigen::Vector2d(2, -1);
    EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), std::invalid_argument);
}

struct QuadraticLocalEnergy : LocalEnergy3D {
    Eigen::Vector2d coefficients{0.2, -0.3};
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
        return (1 + xi.x()) * Eigen::Matrix2d::Identity();
    }
    double eval_feasible_step(const Eigen::Vector3d& xi, const Eigen::Ref<const Eigen::VectorXd>&) const {
        return 0.2 + 0.3 * xi.z();
    }
};

TEST(VolumetricElement3D, IntegratesNonMaterialLocalEnergyWithItsOwnDofCountAndStep) {
    ReferenceField X;
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

TEST(VolumetricElement3D, OwnsIntegrationWeightsWithoutRetainingReferenceField) {
    ReferenceField X;
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
    const auto detached = [] { return VolumetricElement3D<GaussLegendre3D<2>>(ReferenceField{}); }();
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
    const ReferenceField X;
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
    const VolumetricElement3D<GaussLegendre3D<2>> element(ReferenceField{});
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
    const ReferenceField X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    InvalidStepLocalEnergy local;
    for (double bad : {-1., 2., std::numeric_limits<double>::quiet_NaN()}) {
        local.step = bad;
        EXPECT_THROW(element.max_feasible_step(local, Eigen::Vector2d::Zero()), std::logic_error);
    }
}

TEST(VolumetricElement3D, InvertedStatesAndNumericalOverflowThrowDomainError) {
    ReferenceField X;
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(4, 3);
    for (int a = 0; a < 4; ++a)
        u.row(a) = -2 * TetP1Basis::coefficient_info(a).point.transpose();
    EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), std::domain_error);
    EXPECT_THROW(step_displacement<TetP1Basis>(element, X, u, u, Fp, theta), std::domain_error);
    u.setZero();
    u(1, 0) = 1e155;
    EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), std::domain_error);
    u.setZero();
    theta = IdentityField3D(Eigen::Vector2d(0, 1e308));
    Eigen::MatrixXd H;
    EXPECT_DOUBLE_EQ(eval_displacement<TetP1Basis>(element, X, u, Fp, theta), 0);
    EXPECT_THROW(eval_displacement<TetP1Basis>(element, X, u, Fp, theta, nullptr, &H), std::domain_error);

    X.J_X = Eigen::Vector3d(1e-200, 1e100, 1e100).asDiagonal();
    const VolumetricElement3D<TetCentroidQuadrature> anisotropic(X);
    theta = IdentityField3D(Eigen::Vector2d(2, 3));
    EXPECT_THROW(eval_displacement<TetP1Basis>(anisotropic, X, u, Fp, theta, nullptr, &H), std::domain_error);
    Eigen::MatrixXd du = u;
    du(1, 0) = 1e200;
    EXPECT_THROW(eval_displacement<TetP1Basis>(anisotropic, X, du, Fp, theta), std::domain_error);
    EXPECT_THROW(step_displacement<TetP1Basis>(anisotropic, X, u, du, Fp, theta), std::domain_error);

    X.J_X = 10 * Eigen::Matrix3d::Identity();
    const VolumetricElement3D<TetCentroidQuadrature> large_volume(X);
    for (int a = 0; a < 4; ++a)
        u.row(a) = 10 * TetP1Basis::coefficient_info(a).point.transpose();
    theta = IdentityField3D(Eigen::Vector2d(0, 1e307));
    EXPECT_THROW(eval_displacement<TetP1Basis>(large_volume, X, u, Fp, theta), std::domain_error);
}

} // namespace

} // namespace litepgo::fem::test

TEST(VolumetricElementViews, IntegratesIntoPaddedStridedOutputsWithoutTouchingNeighbors) {
    using namespace litepgo;
    using namespace litepgo::fem;
    Eigen::Matrix<double, 4, 3> nodes;
    for (int i = 0; i < 4; ++i)
        nodes.row(i) = TetP1Basis::coefficient_info(i).point.transpose();
    const InterpolatedField3D<TetP1Basis> X(nodes);
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    const IdentityField3D F((1.1 * Eigen::Matrix3d::Identity()).eval().reshaped());
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta(Eigen::Vector2d(2, 3));
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> local(F, Fp, theta);
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

#include "litepgo/fem/material/volumetric/volumetric_material_force_local_response.hpp"

TEST(VolumetricElementViews, ResponseWritesPaddedRectangularBatchBlocks) {
    using namespace litepgo;
    using namespace litepgo::fem;
    Eigen::Matrix<double, 4, 3> nodes;
    for (int i = 0; i < 4; ++i)
        nodes.row(i) = TetP1Basis::coefficient_info(i).point.transpose();
    const InterpolatedField3D<TetP1Basis> X(nodes);
    const VolumetricElement3D<TetCentroidQuadrature> element(X);
    const IdentityField3D F((1.1 * Eigen::Matrix3d::Identity()).eval().reshaped());
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const IdentityField3D theta(Eigen::Vector2d(2, 3));
    const VolumetricMaterialForceLocalResponse3D<NeoHookean> local(F, Fp, theta);
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
    const InterpolatedField3D<HexQ1Basis> X(nodes);
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
