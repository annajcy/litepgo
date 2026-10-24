#include "litepgo/fem/field_interpolation.hpp"
#include "litepgo/fem/gauss_legendre_1d.hpp"
#include "litepgo/fem/gauss_legendre_3d.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/tensor_product_quadrature.hpp"
#include "litepgo/fem/volume_quadrature_cache.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

Eigen::Matrix<double, 8, 3> corners() {
    Eigen::Matrix<double, 8, 3> X;
    X << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return X;
}

Eigen::Matrix<double, 8, 3> affine_field(const Eigen::Matrix<double, 8, 3>& X, const Eigen::Matrix3d& A,
                                         const Eigen::Vector3d& offset) {
    Eigen::Matrix<double, 8, 3> coefficients;
    for (int i = 0; i < 8; ++i) {
        coefficients.row(i) = (A * X.row(i).transpose() + offset).transpose();
    }
    return coefficients;
}

constexpr double alpha = 0.4;
constexpr double beta = 0.2;

Eigen::Vector3d warped_position(const Eigen::Vector3d& xi) {
    return {xi.x() + alpha * xi.y() * xi.z(), xi.y() + beta * xi.x() * xi.z(), xi.z()};
}

Eigen::Vector3d warped_parameter(const Eigen::Vector3d& X) {
    const double determinant = 1.0 - alpha * beta * X.z() * X.z();
    return {(X.x() - alpha * X.z() * X.y()) / determinant, (X.y() - beta * X.z() * X.x()) / determinant, X.z()};
}

Eigen::Matrix<double, 8, 3> warped_geometry() {
    Eigen::Matrix<double, 8, 3> X = corners();
    for (int i = 0; i < 8; ++i) {
        X.row(i) = warped_position(X.row(i).transpose()).transpose();
    }
    return X;
}

template <int Components> void check_coefficient_derivatives() {
    SCOPED_TRACE(Components);
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> mapping(corners() *
                                                                        Eigen::Vector3d(2.0, 3.0, 4.0).asDiagonal());
    Eigen::Matrix<double, 8, Components> coefficients;
    for (int i = 0; i < coefficients.rows(); ++i) {
        for (int c = 0; c < Components; ++c) {
            coefficients(i, c) = std::sin(0.37 * (i * Components + c + 1));
        }
    }
    const auto point = mapping.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4));
    const litepgo::FieldInterpolation field(coefficients);
    const auto sample_value = litepgo::FieldInterpolation(coefficients).value(point);
    const auto sample_gradient = litepgo::FieldInterpolation(coefficients).gradient(point);
    const Eigen::Matrix<double, (8 * Components), 1> flattened = coefficients.template reshaped<Eigen::RowMajor>();
    EXPECT_TRUE((field.value_matrix(point) * flattened).isApprox(sample_value, 1e-14));
    const Eigen::Map<const Eigen::Matrix<double, 3 * Components, 1>> gradient_vector(sample_gradient.data());
    EXPECT_TRUE((field.gradient_matrix(point) * flattened).isApprox(gradient_vector, 1e-14));

    constexpr double h = 1e-4;
    for (int i = 0; i < coefficients.rows(); ++i) {
        for (int c = 0; c < Components; ++c) {
            auto plus = coefficients;
            auto minus = coefficients;
            plus(i, c) += h;
            minus(i, c) -= h;
            const auto sample_plus_value = litepgo::FieldInterpolation(plus).value(point);
            const auto sample_plus_gradient = litepgo::FieldInterpolation(plus).gradient(point);
            const auto sample_minus_value = litepgo::FieldInterpolation(minus).value(point);
            const auto sample_minus_gradient = litepgo::FieldInterpolation(minus).gradient(point);
            const int column = i * Components + c;
            EXPECT_TRUE(((sample_plus_value - sample_minus_value) / (2 * h))
                            .isApprox(field.value_matrix(point).col(column), 1e-9))
                << "coefficient (" << i << ", " << c << ")";
            const Eigen::Map<const Eigen::Matrix<double, 3 * Components, 1>> gradient_plus(sample_plus_gradient.data());
            const Eigen::Map<const Eigen::Matrix<double, 3 * Components, 1>> gradient_minus(
                sample_minus_gradient.data());
            EXPECT_TRUE(
                ((gradient_plus - gradient_minus) / (2 * h)).isApprox(field.gradient_matrix(point).col(column), 1e-9))
                << "coefficient (" << i << ", " << c << ")";
        }
    }
}

} // namespace

TEST(FieldInterpolation, SeparateRestGeometryDisplacementPlasticDisplacementAndMaterialParameters) {
    Eigen::Matrix3d J;
    J << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1);
    const Eigen::Matrix<double, 8, 3> X = (corners() * J.transpose()).rowwise() + origin.transpose();
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> mapping(X);

    Eigen::Matrix3d A;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    const Eigen::Vector3d a_offset(-0.3, 0.4, 0.7);
    const auto a = affine_field(X, A, a_offset);
    const Eigen::Matrix3d P = (Eigen::Matrix3d() << 0.1, 0.02, 0, 0.02, -0.05, 0, 0, 0, 0.03).finished();
    const Eigen::Vector3d b_offset(0.6, -0.2, 0.3);
    const auto b = affine_field(X, P, b_offset);
    const litepgo::FieldInterpolation displacement(a);
    const litepgo::FieldInterpolation plastic_displacement(b);

    Eigen::Matrix<double, 2, 3> material_gradient;
    material_gradient << 0.4, -0.1, 0.2, -0.3, 0.5, 0.1;
    const Eigen::Vector2d material_offset(2.0, 3.0);
    Eigen::Matrix<double, 8, 2> theta;
    for (int i = 0; i < 8; ++i) {
        theta.row(i) = (material_gradient * X.row(i).transpose() + material_offset).transpose();
    }
    const litepgo::FieldInterpolation material_parameters(theta);

    for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(0.83, 0.17, 0.61)}) {
        const Eigen::Vector3d position = origin + J * xi;
        const auto point = mapping.evaluate(xi);
        const auto u_value = displacement.value(point);
        const auto u_gradient = displacement.gradient(point);
        const auto up_value = plastic_displacement.value(point);
        const auto up_gradient = plastic_displacement.gradient(point);
        const auto parameters_value = material_parameters.value(point);
        const auto parameters_gradient = material_parameters.gradient(point);
        EXPECT_TRUE(u_value.isApprox(A * position + a_offset, 1e-13));
        EXPECT_TRUE(u_gradient.isApprox(A, 1e-13));
        EXPECT_TRUE(up_value.isApprox(P * position + b_offset, 1e-13));
        EXPECT_TRUE(up_gradient.isApprox(P, 1e-13));
        EXPECT_TRUE((Eigen::Matrix3d::Identity() + u_gradient).isApprox(Eigen::Matrix3d::Identity() + A, 1e-13));
        EXPECT_TRUE((Eigen::Matrix3d::Identity() + up_gradient).isApprox(Eigen::Matrix3d::Identity() + P, 1e-13));
        EXPECT_TRUE(parameters_value.isApprox(material_gradient * position + material_offset, 1e-13));
        EXPECT_TRUE(parameters_gradient.isApprox(material_gradient, 1e-13));
    }
}

TEST(FieldInterpolation, ValueAndGradientMatricesDifferentiateLocalCoefficients) {
    check_coefficient_derivatives<1>();
    check_coefficient_derivatives<2>();
    check_coefficient_derivatives<3>();
}

TEST(FieldInterpolation, AcceptsEitherMatrixStorageOrder) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> mapping(corners());
    const Eigen::Matrix<double, 8, 3, Eigen::RowMajor> coefficients = corners();
    const auto point = mapping.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4));
    const auto sample_value = litepgo::FieldInterpolation(coefficients).value(point);
    const auto sample_gradient = litepgo::FieldInterpolation(coefficients).gradient(point);
    EXPECT_TRUE(sample_value.isApprox(point.position, 1e-14));
    EXPECT_TRUE(sample_gradient.isIdentity(1e-14));
}

TEST(FieldInterpolation, GradientMatchesPhysicalFiniteDifferencesOnWarpedHex) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> mapping(warped_geometry());
    Eigen::Matrix<double, 8, 3> b;
    for (int i = 0; i < b.rows(); ++i) {
        for (int c = 0; c < b.cols(); ++c) {
            b(i, c) = std::sin(0.37 * (3 * i + c + 1));
        }
    }
    const litepgo::FieldInterpolation plastic_displacement(b);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const auto sample_gradient = plastic_displacement.gradient(mapping.evaluate(xi));
    const Eigen::Vector3d position = warped_position(xi);
    constexpr double h = 1e-6;
    for (int d = 0; d < 3; ++d) {
        Eigen::Vector3d plus = position;
        Eigen::Vector3d minus = position;
        plus[d] += h;
        minus[d] -= h;
        const Eigen::Vector3d value_plus = plastic_displacement.value(mapping.evaluate(warped_parameter(plus)));
        const Eigen::Vector3d value_minus = plastic_displacement.value(mapping.evaluate(warped_parameter(minus)));
        EXPECT_TRUE(((value_plus - value_minus) / (2 * h)).isApprox(sample_gradient.col(d), 3e-9));
    }
}

TEST(FieldInterpolation, DirectAndCachedGeometryPointsGiveTheSameFields) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> mapping(warped_geometry());
    const litepgo::VolumeQuadratureCache cache(mapping, litepgo::GaussLegendre3D<2>{});
    Eigen::Matrix<double, 8, 3> a;
    Eigen::Matrix<double, 8, 3> b;
    Eigen::Matrix<double, 8, 2> theta;
    for (int i = 0; i < a.rows(); ++i) {
        for (int c = 0; c < a.cols(); ++c) {
            a(i, c) = std::sin(0.37 * (3 * i + c + 1));
            b(i, c) = std::cos(0.23 * (3 * i + c + 1));
        }
    }
    for (int i = 0; i < theta.rows(); ++i) {
        for (int c = 0; c < theta.cols(); ++c) {
            theta(i, c) = 2.0 + 0.1 * (2 * i + c);
        }
    }
    const litepgo::FieldInterpolation displacement(a);
    const litepgo::FieldInterpolation plastic_displacement(b);
    const litepgo::FieldInterpolation material_parameters(theta);

    for (int q = 0; q < cache.num_quadrature_points(); ++q) {
        const Eigen::Vector3d xi = litepgo::GaussLegendre3D<2>::points().row(q).transpose();
        const auto direct = mapping.evaluate(xi);
        const auto& cached = cache.sample(q).point;
        const auto direct_u_value = displacement.value(direct);
        const auto direct_u_gradient = displacement.gradient(direct);
        const auto cached_u_value = displacement.value(cached);
        const auto cached_u_gradient = displacement.gradient(cached);
        EXPECT_TRUE(cached_u_value.isApprox(direct_u_value, 1e-14));
        EXPECT_TRUE(cached_u_gradient.isApprox(direct_u_gradient, 1e-14));
        const auto direct_up_value = plastic_displacement.value(direct);
        const auto direct_up_gradient = plastic_displacement.gradient(direct);
        const auto cached_up_value = plastic_displacement.value(cached);
        const auto cached_up_gradient = plastic_displacement.gradient(cached);
        EXPECT_TRUE(cached_up_value.isApprox(direct_up_value, 1e-14));
        EXPECT_TRUE(cached_up_gradient.isApprox(direct_up_gradient, 1e-14));
        const auto direct_theta_value = material_parameters.value(direct);
        const auto direct_theta_gradient = material_parameters.gradient(direct);
        const auto cached_theta_value = material_parameters.value(cached);
        const auto cached_theta_gradient = material_parameters.gradient(cached);
        EXPECT_TRUE(cached_theta_value.isApprox(direct_theta_value, 1e-14));
        EXPECT_TRUE(cached_theta_gradient.isApprox(direct_theta_gradient, 1e-14));
    }
}

TEST(FieldInterpolation, CoefficientsAreCopiedAndTranslationHasZeroGradient) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> mapping(corners());
    const Eigen::Vector3d translation(0.7, -0.2, 1.3);
    Eigen::Matrix<double, 8, 3> coefficients;
    for (int i = 0; i < 8; ++i) {
        coefficients.row(i) = translation.transpose();
    }
    const litepgo::FieldInterpolation field(coefficients);
    coefficients.setConstant(std::numeric_limits<double>::quiet_NaN());
    const auto sample_value = field.value(mapping.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4)));
    const auto sample_gradient = field.gradient(mapping.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4)));
    EXPECT_TRUE(sample_value.isApprox(translation, 1e-14));
    EXPECT_TRUE(sample_gradient.isZero(1e-14));
}

TEST(FieldInterpolation, RejectsNonFiniteCoefficients) {
    Eigen::Matrix<double, 8, 3> coefficients = Eigen::Matrix<double, 8, 3>::Zero();
    coefficients(1, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((litepgo::FieldInterpolation(coefficients)), std::invalid_argument);
}

TEST(FieldInterpolation, ValueAndGradientEvaluateAndValidateIndependently) {
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> mapping(corners());
    const auto point = mapping.evaluate(Eigen::Vector3d(0.2, 0.3, 0.4));
    const litepgo::FieldInterpolation field(corners());
    auto bad_values = point;
    bad_values.basis_values.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(field.value(bad_values), std::invalid_argument);
    EXPECT_TRUE(field.gradient(bad_values).isApprox(field.gradient(point)));
    EXPECT_TRUE((field.gradient_matrix(bad_values).isApprox(field.gradient_matrix(point))));

    auto bad_gradients = point;
    bad_gradients.basis_gradients.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(field.gradient(bad_gradients), std::invalid_argument);
    EXPECT_TRUE(field.value(bad_gradients).isApprox(field.value(point)));
    EXPECT_TRUE((field.value_matrix(bad_gradients).isApprox(field.value_matrix(point))));
}

TEST(FieldInterpolation, RejectsEmptyCoefficientsAndIncompatiblePointDimensions) {
    EXPECT_THROW(litepgo::FieldInterpolation(Eigen::MatrixXd(0, 3)), std::invalid_argument);
    EXPECT_THROW(litepgo::FieldInterpolation(Eigen::MatrixXd(8, 0)), std::invalid_argument);
    const litepgo::FieldInterpolation field(corners());
    EXPECT_EQ(field.num_dofs(), 24);
    EXPECT_EQ(field.num_components(), 3);
    auto point = litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>(corners()).evaluate({0.2, 0.3, 0.4});
    point.basis_values.conservativeResize(7);
    EXPECT_THROW(field.value(point), std::invalid_argument);
    EXPECT_NO_THROW(field.gradient(point));
    point.basis_gradients.conservativeResize(7, 3);
    EXPECT_THROW(field.gradient(point), std::invalid_argument);
    EXPECT_THROW(field.value_matrix(point), std::invalid_argument);
    EXPECT_THROW(field.gradient_matrix(point), std::invalid_argument);
}
