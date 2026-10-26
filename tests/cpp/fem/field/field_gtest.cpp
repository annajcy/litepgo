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
    EXPECT_DOUBLE_EQ(deformation.value(point), 0);
    EXPECT_EQ(deformation.gradient(point).size(), 9);
    EXPECT_EQ(plastic.hessian(point).rows(), 9);
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
    Eigen::MatrixXd dvalue_dx_local(const RestPointData&) const override {
        return Eigen::MatrixXd::Zero(rows, columns);
    }
};
TEST(Field, ActiveJacobianShapeAndPositiveDofCountAreValidated) {
    auto bad = std::make_shared<BadJacobian>();
    MaterialFields fields{bad, std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()),
                          std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3))};
    RestPointData point{};
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>(fields).gradient(point)), std::invalid_argument);
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>(fields).hessian(point)), std::invalid_argument);
    bad->columns = 9;
    bad->rows = 8;
    EXPECT_THROW((MaterialLocalEnergyF<NeoHookean>(fields).gradient(point)), std::invalid_argument);
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
} // namespace
