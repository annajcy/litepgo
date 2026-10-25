#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "quadratic_parameter_material.hpp"

#include "litepgo/fem/geometry/rest.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
TEST(MaterialLocalEnergy, DirectFieldsBindCorrectDerivativesAndOwnSnapshots) {
    Eigen::Matrix<double, 8, 3> rest;
    for (int i = 0; i < 8; ++i)
        rest.row(i) = HexQ1Basis::reference_node(i).transpose();
    const Interpolation<HexQ1Basis> geometry(rest);
    const auto point = litepgo::rest_point(geometry, {0.2, 0.3, 0.4});

    Eigen::Matrix3d F, Fp;
    F << 1.2, 0.1, -0.03, 0.04, 0.9, 0.02, -0.01, 0.04, 1.1;
    Fp << 0.95, -0.02, 0.05, 0.01, 1.1, -0.04, 0.03, 0.01, 1.05;
    MaterialFields fields{std::make_shared<IdentityField<Eigen::Matrix3d>>(F),
                          std::make_shared<IdentityField<Eigen::Matrix3d>>(Fp),
                          std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3))};
    const MaterialLocalEnergyF<NeoHookean> deformation(fields);
    const MaterialLocalEnergyFp<NeoHookean> plastic(fields);
    const MaterialLocalEnergyTheta<NeoHookean> parameters(fields);
    for (const LocalEnergy* energy :
         {static_cast<const LocalEnergy*>(&deformation), static_cast<const LocalEnergy*>(&plastic),
          static_cast<const LocalEnergy*>(&parameters)}) {
        EXPECT_TRUE(energy->has_gradient());
        EXPECT_TRUE(energy->has_hessian());
    }

    const MaterialFields nodal_fields{
        std::make_shared<DeformationGradientField<litepgo::HexQ1Basis>>(Eigen::Matrix<double, 8, 3>::Zero()), fields.Fp,
        fields.theta};
    const MaterialLocalEnergyF<NeoHookean> nodal_deformation(nodal_fields);
    const MaterialLocalEnergyFp<NeoHookean> nodal_plastic(nodal_fields);
    const MaterialLocalEnergyTheta<NeoHookean> nodal_parameters(nodal_fields);
    EXPECT_EQ(nodal_deformation.num_dofs(), 24);
    EXPECT_EQ(nodal_plastic.num_dofs(), 9);
    EXPECT_EQ(nodal_parameters.num_dofs(), 2);
    EXPECT_EQ(nodal_deformation.gradient(point).size(), 24);
    EXPECT_EQ(nodal_plastic.hessian(point).rows(), 9);
    EXPECT_EQ(nodal_parameters.hessian(point).rows(), 2);
    EXPECT_THROW(nodal_deformation.value(RestPointData{}), std::invalid_argument);
    EXPECT_THROW(nodal_plastic.gradient(RestPointData{}), std::invalid_argument);
    EXPECT_THROW(nodal_parameters.hessian(RestPointData{}), std::invalid_argument);
    const auto inputs = fields.evaluate(point);
    const double expected = NeoHookean::value(inputs);
    EXPECT_DOUBLE_EQ(deformation.value(point), expected);
    EXPECT_DOUBLE_EQ(plastic.value(point), expected);
    EXPECT_DOUBLE_EQ(parameters.value(point), expected);
    EXPECT_DOUBLE_EQ(deformation.value(RestPointData{}), expected);
    EXPECT_EQ(plastic.gradient(RestPointData{}), plastic.gradient(point));
    EXPECT_EQ(parameters.hessian(RestPointData{}), parameters.hessian(point));
    const auto stress = NeoHookean::gradient_F(inputs);
    const auto plastic_stress = NeoHookean::gradient_Fp(inputs);
    EXPECT_TRUE(
        deformation.gradient(point).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(stress.data()), 1e-13));
    EXPECT_TRUE(
        plastic.gradient(point).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(plastic_stress.data()), 1e-13));
    EXPECT_EQ(parameters.gradient(point), NeoHookean::gradient_theta(inputs));
    EXPECT_EQ(deformation.hessian(point), NeoHookean::hessian_F(inputs));
    EXPECT_EQ(plastic.hessian(point), NeoHookean::hessian_Fp(inputs));
    EXPECT_EQ(parameters.hessian(point), NeoHookean::hessian_theta(inputs));
    fields.F = std::make_shared<IdentityField<Eigen::Matrix3d>>(1.2 * F);
    fields.Fp = std::make_shared<IdentityField<Eigen::Matrix3d>>(0.9 * Fp);
    fields.theta = std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(4, 5));
    EXPECT_DOUBLE_EQ(deformation.value(point), expected);
    EXPECT_DOUBLE_EQ(plastic.value(point), expected);
    EXPECT_DOUBLE_EQ(parameters.value(point), expected);
    EXPECT_NE((MaterialLocalEnergyF<NeoHookean>(fields).value(point)), expected);
    const auto owned = [&] {
        MaterialFields local{std::make_shared<IdentityField<Eigen::Matrix3d>>(F),
                             std::make_shared<IdentityField<Eigen::Matrix3d>>(Fp),
                             std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3))};
        return MaterialLocalEnergyF<NeoHookean>(std::move(local));
    }();
    EXPECT_DOUBLE_EQ(owned.value(point), expected);
}
} // namespace

namespace {
struct MappedParameters : AffineField<Eigen::VectorXd> {
    Eigen::Vector4d coefficients;
    Eigen::Matrix<double, 3, 4> J;
    explicit MappedParameters(const Eigen::Vector4d& x) : coefficients(x) {
        J << 1, 0.2, 0, -0.3, 0, 2, 0.4, 0, -0.1, 0, 1.5, 0.6;
    }
    int num_dofs() const override {
        return 4;
    }
    Eigen::VectorXd value(const RestPointData&) const override {
        return J * coefficients;
    }
    Eigen::MatrixXd jacobian(const RestPointData&) const override {
        return J;
    }
};
} // namespace

TEST(MaterialLocalEnergy, AnotherMaterialUsesItsOwnParameterCountAndOnlyActiveDerivatives) {
    // This material has no F/Fp derivative methods and has three parameters, unlike NeoHookean.
    using Energy = MaterialLocalEnergyTheta<QuadraticParameterMaterial<3>>;
    MaterialFields fields{std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()),
                          std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()), nullptr};
    const auto bind = [&](const Eigen::Vector4d& x) {
        auto local = fields;
        local.theta = std::make_shared<MappedParameters>(x);
        return Energy(local);
    };
    const Eigen::Vector4d x(0.2, -0.3, 0.5, 0.8);
    const auto energy = bind(x);
    const RestPointData point{};
    const auto g = energy.gradient(point);
    const auto H = energy.hessian(point);
    Eigen::VectorXd accumulated_g = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd accumulated_h = Eigen::MatrixXd::Zero(4, 4);
    EXPECT_DOUBLE_EQ(energy.accumulate(point, 0.3, &accumulated_g, &accumulated_h), 0.3 * energy.value(point));
    EXPECT_TRUE(accumulated_g.isApprox(0.3 * g, 1e-13));
    EXPECT_TRUE(accumulated_h.isApprox(0.3 * H, 1e-13));
    EXPECT_EQ(energy.num_dofs(), 4);
    EXPECT_EQ(g.size(), 4);
    EXPECT_EQ(H.rows(), 4);
    EXPECT_GT(H.norm(), 1);
    constexpr double h = 1e-4;
    for (int j = 0; j < 4; ++j) {
        auto plus = x, minus = x;
        plus[j] += h;
        minus[j] -= h;
        EXPECT_NEAR(g[j], (bind(plus).value(point) - bind(minus).value(point)) / (2 * h), 1e-10);
        EXPECT_TRUE(H.col(j).isApprox((bind(plus).gradient(point) - bind(minus).gradient(point)) / (2 * h), 1e-10));
    }
    fields.theta = std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(1, 2));
    const Energy incompatible(fields);
    EXPECT_THROW(incompatible.value(point), std::invalid_argument);
    EXPECT_THROW(incompatible.gradient(point), std::invalid_argument);
    EXPECT_THROW(incompatible.hessian(point), std::invalid_argument);
}

namespace {
template <class T> struct CountingMaterialField final : litepgo::AffineField<T> {
    T data;
    mutable int value_calls = 0, jacobian_calls = 0;
    explicit CountingMaterialField(T value) : data(std::move(value)) {}
    int num_dofs() const override {
        return static_cast<int>(data.size());
    }
    T value(const litepgo::RestPointData&) const override {
        ++value_calls;
        return data;
    }
    Eigen::MatrixXd jacobian(const litepgo::RestPointData&) const override {
        ++jacobian_calls;
        return Eigen::MatrixXd::Identity(data.size(), data.size());
    }
    void reset() const {
        value_calls = jacobian_calls = 0;
    }
};
} // namespace

TEST(MaterialLocalEnergy, AccumulationReusesFieldValuesAndActiveJacobianForAllActiveFields) {
    const auto F = std::make_shared<CountingMaterialField<Eigen::Matrix3d>>(1.1 * Eigen::Matrix3d::Identity());
    const auto Fp = std::make_shared<CountingMaterialField<Eigen::Matrix3d>>(0.95 * Eigen::Matrix3d::Identity());
    const auto theta = std::make_shared<CountingMaterialField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3));
    const litepgo::MaterialFields fields{F, Fp, theta};
    const litepgo::MaterialLocalEnergyF<litepgo::NeoHookean> deformation(fields);
    const litepgo::MaterialLocalEnergyFp<litepgo::NeoHookean> plastic(fields);
    const litepgo::MaterialLocalEnergyTheta<litepgo::NeoHookean> parameters(fields);
    const litepgo::RestPointData point{};
    const litepgo::LocalEnergy* energies[]{&deformation, &plastic, &parameters};
    for (const auto* energy : energies) {
        const int n = energy->num_dofs();
        for (int request = 0; request < 4; ++request) {
            const bool want_g = request & 1, want_h = request & 2;
            const double weight = request == 3 ? -0.4 : 0.7;
            Eigen::VectorXd g = Eigen::VectorXd::Constant(n, 2);
            Eigen::MatrixXd h = Eigen::MatrixXd::Constant(n, n, 3);
            const double expected = weight * energy->value(point);
            Eigen::VectorXd expected_g = g;
            Eigen::MatrixXd expected_h = h;
            if (want_g)
                expected_g += weight * energy->gradient(point);
            if (want_h)
                expected_h += weight * energy->hessian(point);
            F->reset();
            Fp->reset();
            theta->reset();
            EXPECT_DOUBLE_EQ(energy->accumulate(point, weight, want_g ? &g : nullptr, want_h ? &h : nullptr), expected);
            EXPECT_TRUE(g.isApprox(expected_g, 1e-13));
            EXPECT_TRUE(h.isApprox(expected_h, 1e-13));
            EXPECT_EQ(F->value_calls, 1);
            EXPECT_EQ(Fp->value_calls, 1);
            EXPECT_EQ(theta->value_calls, 1);
            EXPECT_EQ(F->jacobian_calls + Fp->jacobian_calls + theta->jacobian_calls, want_g || want_h ? 1 : 0);
        }
        Eigen::VectorXd g = Eigen::VectorXd::Constant(n, 7);
        EXPECT_THROW(energy->accumulate(point, std::numeric_limits<double>::infinity(), &g, nullptr),
                     std::invalid_argument);
        EXPECT_EQ(g, Eigen::VectorXd::Constant(n, 7));
    }
}
