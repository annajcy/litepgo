#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_value_field.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

namespace {
using namespace litepgo;
struct ValueOnlyParameters final : Field<Eigen::VectorXd> {
    Eigen::VectorXd value(const RestPointData&) const override {
        return Eigen::Vector2d(2, 3);
    }
};
struct ValueOnlyDeformation final : Field<Eigen::Matrix3d> {
    Eigen::Matrix3d value(const RestPointData&) const override {
        return Eigen::Matrix3d::Identity();
    }
};
static_assert(std::is_base_of_v<AffineField<Eigen::Matrix3d>, DeformationGradientField<litepgo::HexQ1Basis>>);
static_assert(std::is_base_of_v<AffineField<Eigen::VectorXd>, InterpolatedValueField<litepgo::HexQ1Basis>>);
static_assert(std::is_base_of_v<AffineField<Eigen::Matrix3d>, IdentityField<Eigen::Matrix3d>>);
static_assert(std::is_base_of_v<AffineField<Eigen::VectorXd>, IdentityField<Eigen::VectorXd>>);
static_assert(!std::is_base_of_v<AffineField<Eigen::VectorXd>, ValueOnlyParameters>);

TEST(Field, InactiveFieldsNeedOnlyValuesAndActiveCapabilitiesAreCheckedAtConstruction) {
    MaterialFields fields{std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()),
                          std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()),
                          std::make_shared<ValueOnlyParameters>()};
    RestPointData point{};
    MaterialLocalEnergyF<NeoHookean> deformation(fields);
    MaterialLocalEnergyFp<NeoHookean> plastic(fields);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(9);
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(9, 9);
    EXPECT_DOUBLE_EQ(deformation.accumulate(point, 1, &g, nullptr), 0);
    EXPECT_DOUBLE_EQ(plastic.accumulate(point, 1, nullptr, &h), 0);
    EXPECT_TRUE(g.isZero());
    EXPECT_TRUE(h.allFinite());
    EXPECT_THROW((MaterialLocalEnergyTheta<NeoHookean>{fields}), std::invalid_argument);
    fields.F = std::make_shared<ValueOnlyDeformation>();
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>{fields}), std::invalid_argument);
    EXPECT_NO_THROW((MaterialLocalEnergyFp<NeoHookean>{fields}));
    fields.Fp.reset();
    EXPECT_THROW(fields.evaluate(point), std::invalid_argument);
    EXPECT_THROW((MaterialLocalEnergyFp<NeoHookean>{fields}), std::invalid_argument);
}

struct BadJacobian final : AffineField<Eigen::Matrix3d> {
    int n = 9;
    int rows = 9;
    int columns = 8;
    int num_dofs() const override {
        return n;
    }
    Eigen::Matrix3d value(const RestPointData&) const override {
        return Eigen::Matrix3d::Identity();
    }
    Eigen::MatrixXd jacobian(const RestPointData&) const override {
        return Eigen::MatrixXd::Zero(rows, columns);
    }
};
TEST(Field, ActiveJacobianShapeAndPositiveDofCountAreValidated) {
    auto bad = std::make_shared<BadJacobian>();
    MaterialFields fields{bad, std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()),
                          std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3))};
    RestPointData point{};
    Eigen::VectorXd g = Eigen::VectorXd::Zero(9);
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(9, 9);
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>(fields).accumulate(point, 1, &g, nullptr)), std::invalid_argument);
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>(fields).accumulate(point, 1, nullptr, &h)), std::invalid_argument);
    bad->columns = 9;
    bad->rows = 8;
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>(fields).accumulate(point, 1, &g, nullptr)), std::invalid_argument);
    bad->n = 0;
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>{fields}), std::invalid_argument);
}
struct InvalidParameters final : Field<Eigen::VectorXd> {
    Eigen::VectorXd parameters;
    Eigen::VectorXd value(const RestPointData&) const override {
        return parameters;
    }
};
TEST(Field, MaterialFieldsRejectEmptyAndNonFiniteRuntimeOutputs) {
    auto parameters = std::make_shared<InvalidParameters>();
    MaterialFields fields{std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()),
                          std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()), parameters};
    EXPECT_THROW(fields.evaluate(RestPointData{}), std::invalid_argument);
    parameters->parameters = Eigen::Vector2d(2, std::numeric_limits<double>::infinity());
    EXPECT_THROW(fields.evaluate(RestPointData{}), std::invalid_argument);
}

struct MappedMatrixField final : AffineField<Eigen::Matrix2d> {
    Eigen::Matrix<double, 4, 3> J;
    Eigen::Vector3d coefficients = Eigen::Vector3d::Zero();
    Eigen::Vector4d offset = Eigen::Vector4d::Zero();
    explicit MappedMatrixField(const Eigen::Matrix<double, 4, 3>& mapping) : J(mapping) {}
    int num_dofs() const override {
        return 3;
    }
    Eigen::Matrix2d value(const RestPointData&) const override {
        const Eigen::Vector4d result = J * coefficients + offset;
        return Eigen::Map<const Eigen::Matrix2d>(result.data());
    }
    Eigen::MatrixXd jacobian(const RestPointData&) const override {
        return J;
    }
};

TEST(Field, DefaultPullbackMatchesFiniteDifferencesWithColumnMajorPacking) {
    Eigen::Matrix<double, 4, 3> J;
    J << 1, 0.2, -0.3, 0.1, 2, 0.4, -0.2, 0.3, 0.8, 0.7, -0.4, 1.2;
    MappedMatrixField field(J);
    field.offset << 0.3, 0.1, -0.2, 0.4;
    Eigen::Matrix4d H = Eigen::Vector4d(1, 2, 3, 4).asDiagonal();
    H(0, 3) = H(3, 0) = 0.2;
    const RestPointData point{};
    const auto energy = [&](const Eigen::Vector3d& x) {
        auto local = field;
        local.coefficients = x;
        const Eigen::Vector4d y = local.value(point).reshaped();
        return 0.5 * y.dot(H * y);
    };
    const auto gradient = [&](const Eigen::Vector3d& x) {
        auto local = field;
        local.coefficients = x;
        const Eigen::Vector4d g = H * local.value(point).reshaped();
        const Eigen::Matrix<double, 2, 2, Eigen::RowMajor> matrix = Eigen::Map<const Eigen::Matrix2d>(g.data());
        Eigen::VectorXd result = Eigen::Vector3d::Zero();
        local.accumulate_derivatives(point, matrix, Eigen::MatrixXd{}, 1, &result, nullptr);
        return result;
    };
    const Eigen::Vector3d a(0.2, -0.3, 0.5);
    const auto g = gradient(a);
    Eigen::MatrixXd hessian = Eigen::Matrix3d::Zero();
    field.accumulate_derivatives(point, Eigen::MatrixXd{}, H, 1, nullptr, &hessian);
    for (double h : {1e-3, 1e-4})
        for (int j = 0; j < 3; ++j) {
            auto plus = a, minus = a;
            plus[j] += h;
            minus[j] -= h;
            EXPECT_NEAR(g[j], (energy(plus) - energy(minus)) / (2 * h), 1e-10);
            EXPECT_TRUE(hessian.col(j).isApprox((gradient(plus) - gradient(minus)) / (2 * h), 1e-10));
        }
    Eigen::MatrixXd scaled_hessian = Eigen::Matrix3d::Zero();
    MappedMatrixField(2 * J).accumulate_derivatives(point, Eigen::MatrixXd{}, H + H, 1, nullptr, &scaled_hessian);
    EXPECT_TRUE(scaled_hessian.isApprox(8 * hessian, 1e-13));
}

TEST(Field, DefaultPullbackAccumulatesWeightedStridedGradientsAndNonsymmetricHessians) {
    Eigen::Matrix<double, 4, 3> J;
    J << 1, 0.2, -0.3, 0.1, 2, 0.4, -0.2, 0.3, 0.8, 0.7, -0.4, 1.2;
    const MappedMatrixField field(J);
    Eigen::Matrix4d H = Eigen::Vector4d(1, 2, 3, 4).asDiagonal();
    H(0, 3) = 0.2;
    H(3, 0) = -0.1;
    const Eigen::MatrixXd storage = Eigen::MatrixXd::Random(5, 4);
    const auto gradient = storage.block<2, 2>(1, 1);
    const Eigen::Matrix2d packed = gradient;
    const Eigen::Vector3d expected_g = J.transpose() * packed.reshaped();
    const Eigen::Matrix3d expected_h = J.transpose() * H * J;
    for (double weight : {0.0, 0.3, -0.7})
        for (int request = 0; request < 4; ++request) {
            Eigen::VectorXd g = Eigen::Vector3d::Constant(2);
            Eigen::MatrixXd h = Eigen::Matrix3d::Constant(4);
            field.accumulate_derivatives(RestPointData{}, gradient, H, weight, request & 1 ? &g : nullptr,
                                         request & 2 ? &h : nullptr);
            EXPECT_TRUE(g.isApprox(Eigen::Vector3d::Constant(2) + (request & 1 ? weight : 0) * expected_g, 1e-13));
            EXPECT_TRUE(h.isApprox(Eigen::Matrix3d::Constant(4) + (request & 2 ? weight : 0) * expected_h, 1e-13));
        }
}

TEST(Field, DefaultPullbackRejectsInvalidInputsBeforeWriting) {
    MappedMatrixField field(Eigen::Matrix<double, 4, 3>::Ones());
    const RestPointData point{};
    const Eigen::Vector4d gradient = Eigen::Vector4d::Ones();
    const Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
    Eigen::VectorXd g = Eigen::Vector3d::Constant(7);
    Eigen::MatrixXd h = Eigen::Matrix3d::Constant(9);
    EXPECT_THROW(field.accumulate_derivatives(point, Eigen::Vector3d::Ones(), H, 1, &g, &h), std::invalid_argument);
    EXPECT_THROW(field.accumulate_derivatives(point, gradient, Eigen::Matrix3d::Identity(), 1, &g, &h),
                 std::invalid_argument);
    EXPECT_THROW(field.accumulate_derivatives(point, gradient, H, std::numeric_limits<double>::infinity(), &g, &h),
                 std::invalid_argument);
    EXPECT_EQ(g, Eigen::Vector3d::Constant(7));
    EXPECT_EQ(h, Eigen::Matrix3d::Constant(9));
    g.resize(2);
    h.resize(3, 2);
    EXPECT_THROW(field.accumulate_derivatives(point, gradient, H, 1, &g, nullptr), std::invalid_argument);
    EXPECT_THROW(field.accumulate_derivatives(point, gradient, H, 1, nullptr, &h), std::invalid_argument);
    g = Eigen::Vector3d::Constant(7);
    h = Eigen::Matrix3d::Constant(9);
    field.J(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(field.accumulate_derivatives(point, gradient, H, 1, &g, &h), std::invalid_argument);
    EXPECT_EQ(g, Eigen::Vector3d::Constant(7));
    EXPECT_EQ(h, Eigen::Matrix3d::Constant(9));
}

TEST(Field, DefaultPullbackAppliesWeightWithoutIntermediateOverflowOrUnderflow) {
    struct Scales {
        double jacobian, material, weight, expected_gradient;
    };
    for (const auto scales : {Scales{1e-100, 1e200, 1e150, 1e250}, Scales{1e100, 1e-200, 1e-150, 1e-250}}) {
        Eigen::Matrix<double, 4, 3> J = Eigen::Matrix<double, 4, 3>::Zero();
        J.topRows<3>() = scales.jacobian * Eigen::Matrix3d::Identity();
        const MappedMatrixField field(J);
        const Eigen::Vector4d gradient = Eigen::Vector4d::Constant(scales.material);
        const Eigen::Matrix4d H = scales.material * Eigen::Matrix4d::Identity();
        Eigen::VectorXd g = Eigen::Vector3d::Zero();
        Eigen::MatrixXd h = Eigen::Matrix3d::Zero();
        field.accumulate_derivatives(RestPointData{}, gradient, H, scales.weight, &g, &h);
        EXPECT_TRUE(g.allFinite());
        EXPECT_TRUE(h.allFinite());
        EXPECT_TRUE((g / scales.expected_gradient).isApprox(Eigen::Vector3d::Ones(), 1e-13));
        EXPECT_TRUE((h / scales.weight).isIdentity(1e-13));
    }
}
} // namespace
