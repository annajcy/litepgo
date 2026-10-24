#include "litepgo/fem/gauss_legendre_3d.h"
#include "litepgo/fem/hex_q1_basis.h"
#include "litepgo/fem/interpolated_field.h"
#include "litepgo/fem/volume_quadrature_cache.h"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using Basis = litepgo::HexQ1Basis;
using Mapping = litepgo::VolumeReferenceGeometry<Basis>;
using Quadrature = litepgo::GaussLegendre3D<2>;
using Cache = litepgo::VolumeQuadratureCache<Basis, Quadrature>;
using VectorField = litepgo::InterpolatedField<Basis, 3>;
using MaterialField = litepgo::InterpolatedField<Basis, 2>;
using Geometry = Mapping::GeometryCoefficients;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

template <class B, int C>
concept CanFormField = requires { typename litepgo::InterpolatedField<B, C>; };

static_assert(CanFormField<Basis, 1>);
static_assert(CanFormField<Basis, 3>);
static_assert(!CanFormField<Basis, 0>);
static_assert(!CanFormField<litepgo::LagrangeBasis, 3>);
static_assert(VectorField::num_dofs == 24);
static_assert(MaterialField::num_dofs == 16);
static_assert(
    std::same_as<decltype(std::declval<const VectorField&>().evaluate(std::declval<const Mapping::PointData&>())),
                 VectorField::Sample>);

Geometry corners() {
    Geometry X;
    X << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return X;
}

VectorField::LocalCoefficients affine_field(const Geometry& X, const Mat3& A, const Vec3& offset) {
    VectorField::LocalCoefficients coefficients;
    for (int i = 0; i < 8; ++i) {
        coefficients.segment<3>(3 * i) = A * X.row(i).transpose() + offset;
    }
    return coefficients;
}

constexpr double alpha = 0.4;
constexpr double beta = 0.2;

Vec3 warped_position(const Vec3& xi) {
    return {xi.x() + alpha * xi.y() * xi.z(), xi.y() + beta * xi.x() * xi.z(), xi.z()};
}

Vec3 warped_parameter(const Vec3& X) {
    const double determinant = 1.0 - alpha * beta * X.z() * X.z();
    return {(X.x() - alpha * X.z() * X.y()) / determinant, (X.y() - beta * X.z() * X.x()) / determinant, X.z()};
}

Geometry warped_geometry() {
    Geometry X = corners();
    for (int i = 0; i < 8; ++i) {
        X.row(i) = warped_position(X.row(i).transpose()).transpose();
    }
    return X;
}

} // namespace

TEST(InterpolatedField, SeparateRestGeometryDisplacementPlasticDisplacementAndMaterialParameters) {
    Mat3 J;
    J << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Vec3 origin(1.2, -0.7, 2.1);
    const Geometry X = (corners() * J.transpose()).rowwise() + origin.transpose();
    const Mapping mapping(X);

    Mat3 A;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    const Vec3 a_offset(-0.3, 0.4, 0.7);
    const auto a = affine_field(X, A, a_offset);
    const Mat3 P = (Mat3() << 0.1, 0.02, 0, 0.02, -0.05, 0, 0, 0, 0.03).finished();
    const Vec3 b_offset(0.6, -0.2, 0.3);
    const auto b = affine_field(X, P, b_offset);
    const VectorField displacement(a);
    const VectorField plastic_displacement(b);

    Eigen::Matrix<double, 2, 3> material_gradient;
    material_gradient << 0.4, -0.1, 0.2, -0.3, 0.5, 0.1;
    const Eigen::Vector2d material_offset(2.0, 3.0);
    MaterialField::LocalCoefficients theta;
    for (int i = 0; i < 8; ++i) {
        theta.segment<2>(2 * i) = material_gradient * X.row(i).transpose() + material_offset;
    }
    const MaterialField material_parameters(theta);

    for (const Vec3 xi : {Vec3(0.2, 0.3, 0.4), Vec3(0.83, 0.17, 0.61)}) {
        const Vec3 position = origin + J * xi;
        const auto point = mapping.evaluate(xi);
        const auto u = displacement.evaluate(point);
        const auto up = plastic_displacement.evaluate(point);
        const auto parameters = material_parameters.evaluate(point);
        EXPECT_TRUE(u.value.isApprox(A * position + a_offset, 1e-13));
        EXPECT_TRUE(u.gradient.isApprox(A, 1e-13));
        EXPECT_TRUE(up.value.isApprox(P * position + b_offset, 1e-13));
        EXPECT_TRUE(up.gradient.isApprox(P, 1e-13));
        EXPECT_TRUE((Mat3::Identity() + u.gradient).isApprox(Mat3::Identity() + A, 1e-13));
        EXPECT_TRUE((Mat3::Identity() + up.gradient).isApprox(Mat3::Identity() + P, 1e-13));
        EXPECT_TRUE(parameters.value.isApprox(material_gradient * position + material_offset, 1e-13));
        EXPECT_TRUE(parameters.gradient.isApprox(material_gradient, 1e-13));
    }
}

TEST(InterpolatedField, ValueAndGradientMatricesDifferentiateLocalCoefficients) {
    const Mapping mapping(corners() * Vec3(2.0, 3.0, 4.0).asDiagonal());
    VectorField::LocalCoefficients a;
    for (int i = 0; i < a.size(); ++i) {
        a[i] = std::sin(0.37 * (i + 1));
    }
    const Vec3 xi(0.2, 0.3, 0.4);
    const auto point = mapping.evaluate(xi);
    const auto sample = VectorField(a).evaluate(point);
    EXPECT_TRUE((sample.value_matrix() * a).isApprox(sample.value, 1e-14));
    const Eigen::Map<const Eigen::Matrix<double, 9, 1>> gradient_vector(sample.gradient.data());
    EXPECT_TRUE((sample.gradient_matrix() * a).isApprox(gradient_vector, 1e-14));

    constexpr double h = 1e-6;
    for (int i : {0, 7, 23}) {
        auto perturbed = a;
        perturbed[i] += h;
        const auto changed = VectorField(perturbed).evaluate(point);
        EXPECT_TRUE(((changed.value - sample.value) / h).isApprox(sample.value_matrix().col(i), 1e-9));
        const Eigen::Map<const Eigen::Matrix<double, 9, 1>> changed_gradient(changed.gradient.data());
        EXPECT_TRUE(((changed_gradient - gradient_vector) / h).isApprox(sample.gradient_matrix().col(i), 1e-9));
    }
}

TEST(InterpolatedField, GradientMatchesPhysicalFiniteDifferencesOnWarpedHex) {
    const Mapping mapping(warped_geometry());
    VectorField::LocalCoefficients b;
    for (int i = 0; i < b.size(); ++i) {
        b[i] = std::sin(0.37 * (i + 1));
    }
    const VectorField plastic_displacement(b);
    const Vec3 xi(0.2, 0.3, 0.4);
    const auto sample = plastic_displacement.evaluate(mapping.evaluate(xi));
    const Vec3 position = warped_position(xi);
    constexpr double h = 1e-6;
    for (int d = 0; d < 3; ++d) {
        Vec3 plus = position;
        Vec3 minus = position;
        plus[d] += h;
        minus[d] -= h;
        const Vec3 value_plus = plastic_displacement.evaluate(mapping.evaluate(warped_parameter(plus))).value;
        const Vec3 value_minus = plastic_displacement.evaluate(mapping.evaluate(warped_parameter(minus))).value;
        EXPECT_TRUE(((value_plus - value_minus) / (2 * h)).isApprox(sample.gradient.col(d), 3e-9));
    }
}

TEST(InterpolatedField, DirectAndCachedGeometryPointsGiveTheSameFields) {
    const Mapping mapping(warped_geometry());
    const Cache cache(mapping);
    VectorField::LocalCoefficients a;
    VectorField::LocalCoefficients b;
    MaterialField::LocalCoefficients theta;
    for (int i = 0; i < a.size(); ++i) {
        a[i] = std::sin(0.37 * (i + 1));
        b[i] = std::cos(0.23 * (i + 1));
    }
    for (int i = 0; i < theta.size(); ++i) {
        theta[i] = 2.0 + 0.1 * i;
    }
    const VectorField displacement(a);
    const VectorField plastic_displacement(b);
    const MaterialField material_parameters(theta);

    for (int q = 0; q < Cache::num_quadrature_points; ++q) {
        const Vec3 xi = Quadrature::points().row(q).transpose();
        const auto direct = mapping.evaluate(xi);
        const auto& cached = cache.sample(q).point;
        const auto direct_u = displacement.evaluate(direct);
        const auto cached_u = displacement.evaluate(cached);
        EXPECT_TRUE(cached_u.value.isApprox(direct_u.value, 1e-14));
        EXPECT_TRUE(cached_u.gradient.isApprox(direct_u.gradient, 1e-14));
        const auto direct_up = plastic_displacement.evaluate(direct);
        const auto cached_up = plastic_displacement.evaluate(cached);
        EXPECT_TRUE(cached_up.value.isApprox(direct_up.value, 1e-14));
        EXPECT_TRUE(cached_up.gradient.isApprox(direct_up.gradient, 1e-14));
        const auto direct_theta = material_parameters.evaluate(direct);
        const auto cached_theta = material_parameters.evaluate(cached);
        EXPECT_TRUE(cached_theta.value.isApprox(direct_theta.value, 1e-14));
        EXPECT_TRUE(cached_theta.gradient.isApprox(direct_theta.gradient, 1e-14));
    }
}

TEST(InterpolatedField, CoefficientsAreCopiedAndTranslationHasZeroGradient) {
    const Mapping mapping(corners());
    const Vec3 translation(0.7, -0.2, 1.3);
    VectorField::LocalCoefficients coefficients;
    for (int i = 0; i < 8; ++i) {
        coefficients.segment<3>(3 * i) = translation;
    }
    const VectorField field(coefficients);
    coefficients.setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto sample = field.evaluate(mapping.evaluate(Vec3(0.2, 0.3, 0.4)));
    EXPECT_TRUE(sample.value.isApprox(translation, 1e-14));
    EXPECT_TRUE(sample.gradient.isZero(1e-14));
}

TEST(InterpolatedField, RejectsNonFiniteCoefficients) {
    VectorField::LocalCoefficients coefficients = VectorField::LocalCoefficients::Zero();
    coefficients[4] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((VectorField(coefficients)), std::invalid_argument);
}
