#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/geometry/interpolation.hpp"
#include "litepgo/fem/geometry/rest.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;

template <class B>
concept CanInterpolate = requires { typename Interpolation<B>; };
static_assert(CanInterpolate<HexQ1Basis>);
static_assert(!CanInterpolate<LagrangeBasis>);
static_assert(Interpolation<HexQ1Basis>::num_basis_functions == 8);

Eigen::Matrix<double, 8, 3> corners() {
    Eigen::Matrix<double, 8, 3> X;
    for (int i = 0; i < 8; ++i)
        X.row(i) = HexQ1Basis::reference_node(i).transpose();
    return X;
}

// Static evaluation must not require constructing a basis object.
template <bool InvalidValues> struct InvalidBasis {
    InvalidBasis() = delete;
    static constexpr int num_basis_functions = 8;
    using Sample = HexQ1Basis::Sample;
    static Sample evaluate(const Eigen::Vector3d& xi) {
        auto sample = HexQ1Basis::evaluate(xi);
        if constexpr (InvalidValues)
            sample.values.setConstant(std::numeric_limits<double>::quiet_NaN());
        else
            sample.gradients.setConstant(std::numeric_limits<double>::infinity());
        return sample;
    }
};

template <int Components, int Storage> void check_coefficient_derivatives() {
    SCOPED_TRACE(Components);
    SCOPED_TRACE(Storage);
    Eigen::Matrix<double, 8, Components, Storage> coefficients;
    for (int i = 0; i < 8; ++i)
        for (int c = 0; c < Components; ++c)
            coefficients(i, c) = std::sin(0.37 * (i * Components + c + 1));
    const Interpolation<HexQ1Basis> field(coefficients);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const Eigen::VectorXd flattened = coefficients.template reshaped<Eigen::RowMajor>();
    const auto gradient = field.gradient(xi);
    EXPECT_TRUE((field.value_matrix(xi) * flattened).isApprox(field.value(xi), 1e-14));
    EXPECT_TRUE((field.gradient_matrix(xi) * flattened).isApprox(gradient.reshaped(), 1e-14));
    constexpr double h = 1e-4;
    for (int j = 0; j < field.num_dofs(); ++j) {
        auto plus = coefficients, minus = coefficients;
        plus(j / Components, j % Components) += h;
        minus(j / Components, j % Components) -= h;
        const Interpolation<HexQ1Basis> a(plus), b(minus);
        const Eigen::VectorXd value_fd = (a.value(xi) - b.value(xi)) / (2 * h);
        const Eigen::MatrixXd gradient_fd = (a.gradient(xi) - b.gradient(xi)) / (2 * h);
        EXPECT_TRUE(value_fd.isApprox(field.value_matrix(xi).col(j), 1e-9));
        EXPECT_TRUE(gradient_fd.reshaped().isApprox(field.gradient_matrix(xi).col(j), 1e-9));
    }
}
} // namespace

TEST(Interpolation, ValueAndGradientMatricesDifferentiateRowOrderedDofs) {
    check_coefficient_derivatives<1, Eigen::ColMajor>();
    check_coefficient_derivatives<2, Eigen::ColMajor>();
    check_coefficient_derivatives<2, Eigen::RowMajor>();
    check_coefficient_derivatives<3, Eigen::ColMajor>();
    check_coefficient_derivatives<3, Eigen::RowMajor>();
}

TEST(Interpolation, GeometryAndFieldsUseTheSameParameterInterpolation) {
    Eigen::Matrix3d J, A;
    J << 2, 0.3, 0.1, 0.2, 1.5, 0.1, 0, 0.2, 1.1;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    const Eigen::Vector3d origin(1.2, -0.7, 2.1), offset(-0.3, 0.4, 0.7);
    const Eigen::Matrix<double, 8, 3> X = (corners() * J.transpose()).rowwise() + origin.transpose();
    const Interpolation<HexQ1Basis> geometry(X);
    const Interpolation<HexQ1Basis> displacement((X * A.transpose()).rowwise() + offset.transpose());
    Eigen::Matrix<double, 2, 3> M;
    M << 0.4, -0.1, 0.2, -0.3, 0.5, 0.1;
    const Interpolation<HexQ1Basis> parameters((X * M.transpose()).rowwise() + Eigen::RowVector2d(2, 3));
    for (const Eigen::Vector3d xi : {Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(-0.2, 1.2, 0.4)}) {
        const auto point = rest_point(geometry, xi);
        EXPECT_TRUE(geometry.value(xi).isApprox(origin + J * xi, 1e-13));
        EXPECT_TRUE(geometry.gradient(xi).isApprox(J, 1e-13));
        EXPECT_TRUE(displacement.value(xi).isApprox(A * point.position + offset, 1e-13));
        EXPECT_TRUE(displacement.gradient(xi).isApprox(A * J, 1e-13));
        EXPECT_TRUE((displacement.gradient(xi) * point.inverse_jacobian).isApprox(A, 1e-13));
        EXPECT_TRUE(parameters.value(xi).isApprox(M * point.position + Eigen::Vector2d(2, 3), 1e-13));
        EXPECT_TRUE((parameters.gradient(xi) * point.inverse_jacobian).isApprox(M, 1e-13));
    }
}

TEST(Interpolation, ParameterGradientMatchesFiniteDifferences) {
    Eigen::Matrix<double, 8, 3> coefficients;
    for (int i = 0; i < 8; ++i)
        for (int c = 0; c < 3; ++c)
            coefficients(i, c) = std::sin(0.37 * (3 * i + c + 1));
    const Interpolation<HexQ1Basis> field(coefficients);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    constexpr double h = 1e-6;
    for (int d = 0; d < 3; ++d) {
        Eigen::Vector3d plus = xi, minus = xi;
        plus[d] += h;
        minus[d] -= h;
        EXPECT_TRUE(((field.value(plus) - field.value(minus)) / (2 * h)).isApprox(field.gradient(xi).col(d), 3e-9));
    }
}

TEST(Interpolation, CoefficientsAreCopiedAndTranslationHasZeroGradient) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4), translation(0.7, -0.2, 1.3);
    Eigen::Matrix<double, 8, 3, Eigen::RowMajor> coefficients = translation.transpose().replicate<8, 1>();
    const Interpolation<HexQ1Basis> field(coefficients);
    coefficients.setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(field.value(xi).isApprox(translation, 1e-14));
    EXPECT_TRUE(field.gradient(xi).isZero(1e-14));
    EXPECT_EQ(field.num_dofs(), 24);
    EXPECT_EQ(field.num_components(), 3);
}

TEST(Interpolation, ValueAndGradientValidateIndependently) {
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const Interpolation<InvalidBasis<true>> bad_values(corners());
    EXPECT_THROW(bad_values.value(xi), std::invalid_argument);
    EXPECT_THROW(bad_values.value_matrix(xi), std::invalid_argument);
    EXPECT_TRUE(bad_values.gradient(xi).isIdentity(1e-14));
    EXPECT_NO_THROW(bad_values.gradient_matrix(xi));
    const Interpolation<InvalidBasis<false>> bad_gradients(corners());
    EXPECT_THROW(bad_gradients.gradient(xi), std::invalid_argument);
    EXPECT_THROW(bad_gradients.gradient_matrix(xi), std::invalid_argument);
    EXPECT_TRUE(bad_gradients.value(xi).isApprox(xi, 1e-14));
    EXPECT_NO_THROW(bad_gradients.value_matrix(xi));
}

TEST(Interpolation, RejectsIncompatibleCoefficientsAndNonFiniteInputsOrResults) {
    for (int rows : {0, 7, 9})
        EXPECT_THROW(Interpolation<HexQ1Basis>(Eigen::MatrixXd::Zero(rows, 3)), std::invalid_argument);
    EXPECT_THROW(Interpolation<HexQ1Basis>(Eigen::MatrixXd(8, 0)), std::invalid_argument);
    const Interpolation<HexQ1Basis> field(corners());
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        auto coefficients = corners();
        coefficients(1, 1) = bad;
        EXPECT_THROW((Interpolation<HexQ1Basis>(coefficients)), std::invalid_argument);
        const Eigen::Vector3d xi(bad, 0.3, 0.4);
        EXPECT_THROW(field.value(xi), std::invalid_argument);
        EXPECT_THROW(field.gradient(xi), std::invalid_argument);
        EXPECT_THROW(field.value_matrix(xi), std::invalid_argument);
        EXPECT_THROW(field.gradient_matrix(xi), std::invalid_argument);
    }
    const Interpolation<HexQ1Basis> enormous(Eigen::MatrixXd::Constant(8, 3, std::numeric_limits<double>::max()));
    EXPECT_THROW(enormous.value(Eigen::Vector3d::Constant(2)), std::invalid_argument);
    EXPECT_THROW(enormous.gradient(Eigen::Vector3d::Constant(2)), std::invalid_argument);
}
