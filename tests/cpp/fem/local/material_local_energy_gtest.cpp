#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "quadratic_parameter_material.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <type_traits>

namespace {
using namespace litepgo;
auto coordinates(double scale = 1) {
    Eigen::Matrix<double, 8, 3> X_local;
    for (int a = 0; a < 8; ++a)
        X_local.row(a) = scale * HexQ1Basis::reference_node(a).transpose();
    return InterpolatedField3D<HexQ1Basis>(X_local);
}
struct ValueOnlyField : Field3D {
    int num_dofs() const override {
        return 0;
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        throw std::logic_error("Read-only field");
    }
};
struct ValueOnlyParameters final : ValueOnlyField {
    Eigen::VectorXd data = Eigen::Vector2d(2, 3);
    int num_components() const override {
        return static_cast<int>(data.size());
    }
    Eigen::VectorXd value(const Eigen::Vector3d&) const override {
        return data;
    }
};
struct ValueOnlyF final : ValueOnlyField {
    int num_components() const override {
        return 9;
    }
    Eigen::VectorXd value(const Eigen::Vector3d&) const override {
        return Eigen::Matrix3d::Identity().reshaped();
    }
};
TEST(MaterialLocalEnergy3D, DirectFieldsBindCorrectDerivativesAndBorrowNamedFields) {
    Eigen::Matrix3d F = (Eigen::Matrix3d() << 1.2, 0.1, -0.03, 0.04, 0.9, 0.02, -0.01, 0.04, 1.1).finished();
    const Eigen::Matrix3d Fp = (Eigen::Matrix3d() << 0.95, -0.02, 0.05, 0.01, 1.1, -0.04, 0.03, 0.01, 1.05).finished();
    IdentityField3D F_field(F.reshaped());
    const IdentityField3D Fp_field(Fp.reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    const auto energy = litepgo::MaterialLocalEnergyF3D<NeoHookean>(F_field, Fp_field, theta_field);
    const auto plastic = litepgo::MaterialLocalEnergyFp3D<NeoHookean>(F_field, Fp_field, theta_field);
    const auto parameters = litepgo::MaterialLocalEnergyTheta3D<NeoHookean>(F_field, Fp_field, theta_field);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const MaterialInputs inputs{F, Fp, Eigen::Vector2d(2, 3)};
    const double W = NeoHookean::value(inputs);
    EXPECT_DOUBLE_EQ(energy.value(xi), W);
    EXPECT_DOUBLE_EQ(plastic.value(xi), W);
    EXPECT_DOUBLE_EQ(parameters.value(xi), W);
    EXPECT_TRUE(energy.gradient(xi).isApprox(NeoHookean::dW_dF(inputs).reshaped(), 1e-14));
    EXPECT_TRUE(plastic.gradient(xi).isApprox(NeoHookean::dW_dFp(inputs).reshaped(), 1e-14));
    EXPECT_EQ(parameters.gradient(xi), NeoHookean::dW_dtheta(inputs));
    EXPECT_EQ(energy.hessian(xi), NeoHookean::d2W_dF2(inputs));
    EXPECT_EQ(plastic.hessian(xi), NeoHookean::d2W_dFp2(inputs));
    EXPECT_EQ(parameters.hessian(xi), NeoHookean::d2W_dtheta2(inputs));
    EXPECT_NEAR(energy.accumulate(xi, 8, nullptr, nullptr), 8 * W, 1e-13);
    F.setZero();
    EXPECT_DOUBLE_EQ(energy.value(xi), W);
    F = 1.01 * inputs.F;
    F_field = IdentityField3D(F.reshaped());
    const auto copy = energy;
    EXPECT_DOUBLE_EQ(copy.value(xi), NeoHookean::value({F, Fp, inputs.theta}));
}
TEST(MaterialLocalEnergy3D, InactiveInputsNeedOnlyValuesAndActiveCapabilitiesAreCheckedAtBinding) {
    const ValueOnlyF readonly_F;
    const ValueOnlyParameters readonly_theta;
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const auto energy = litepgo::MaterialLocalEnergyFp3D<NeoHookean>(readonly_F, Fp, readonly_theta);
    EXPECT_DOUBLE_EQ(energy.value({0.2, 0.3, 0.4}), 0);
    EXPECT_EQ(energy.hessian({0.2, 0.3, 0.4}).rows(), 9);
    EXPECT_THROW((litepgo::MaterialLocalEnergyF3D<NeoHookean>(readonly_F, Fp, readonly_theta)), std::invalid_argument);
    EXPECT_THROW((litepgo::MaterialLocalEnergyTheta3D<NeoHookean>(readonly_F, Fp, readonly_theta)),
                 std::invalid_argument);
    ValueOnlyParameters bad;
    bad.data.resize(0);
    const auto empty = litepgo::MaterialLocalEnergyFp3D<NeoHookean>(readonly_F, Fp, bad);
    EXPECT_THROW(empty.value({0.2, 0.3, 0.4}), std::invalid_argument);
    bad.data = Eigen::Vector2d(2, std::numeric_limits<double>::infinity());
    const auto nonfinite = litepgo::MaterialLocalEnergyFp3D<NeoHookean>(readonly_F, Fp, bad);
    EXPECT_THROW(nonfinite.hessian({0.2, 0.3, 0.4}), std::invalid_argument);
}
struct MappedParameters final : Field3D {
    explicit MappedParameters(const Eigen::Vector4d& data) : coeffs(data) {}
    int num_components() const override {
        return 3;
    }
    Eigen::Vector4d coeffs;
    Eigen::Matrix<double, 3, 4> derivative =
        (Eigen::Matrix<double, 3, 4>() << 1, 0.2, 0, -0.3, 0, 2, 0.4, 0, -0.1, 0, 1.5, 0.6).finished();
    int num_dofs() const override {
        return 4;
    }
    Eigen::VectorXd value(const Eigen::Vector3d&) const override {
        return derivative * coeffs;
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        return derivative;
    }
};
TEST(MaterialLocalEnergy3D, AnotherMaterialUsesOnlyItsActiveDerivativesAndOwnParameterCount) {
    const IdentityField3D F(Eigen::Matrix3d::Identity().reshaped());
    const Eigen::Vector4d x(0.2, -0.3, 0.5, 0.8);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const MappedParameters parameters(x);
    const MaterialLocalEnergyTheta3D<QuadraticParameterMaterial<3>> energy(F, F, parameters);
    const auto g = energy.gradient(xi);
    const auto H = energy.hessian(xi);
    for (int j = 0; j < 4; ++j) {
        auto plus = x, minus = x;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const MappedParameters plus_parameters(plus), minus_parameters(minus);
        const MaterialLocalEnergyTheta3D<QuadraticParameterMaterial<3>> plus_energy(F, F, plus_parameters);
        const MaterialLocalEnergyTheta3D<QuadraticParameterMaterial<3>> minus_energy(F, F, minus_parameters);
        EXPECT_NEAR(g[j], (plus_energy.value(xi) - minus_energy.value(xi)) / 2e-4, 1e-10);
        EXPECT_TRUE(H.col(j).isApprox((plus_energy.gradient(xi) - minus_energy.gradient(xi)) / 2e-4, 1e-10));
    }
}

template <class Energy> void check_temporary_inputs_rejected() {
    static_assert(!std::is_constructible_v<Energy, IdentityField3D&&, const IdentityField3D&, const IdentityField3D&>);
    static_assert(
        !std::is_constructible_v<Energy, const IdentityField3D&, const IdentityField3D&&, const IdentityField3D&>);
    static_assert(!std::is_constructible_v<Energy, const IdentityField3D&, const IdentityField3D&, IdentityField3D&&>);
}

TEST(MaterialLocalEnergy3D, AllThreeClassesRejectTemporaryFields) {
    static_assert(std::is_abstract_v<MaterialLocalEnergy3D<NeoHookean>>);
    check_temporary_inputs_rejected<MaterialLocalEnergyF3D<NeoHookean>>();
    check_temporary_inputs_rejected<MaterialLocalEnergyFp3D<NeoHookean>>();
    check_temporary_inputs_rejected<MaterialLocalEnergyTheta3D<NeoHookean>>();
}
struct Counts {
    int values = 0, derivatives = 0;
};
template <class Value> struct CountingField final : Field3D {
    explicit CountingField(Value value) : data(std::move(value)) {}
    int num_components() const override {
        return static_cast<int>(data.size());
    }
    Value data;
    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
    int num_dofs() const override {
        return static_cast<int>(data.size());
    }
    Eigen::VectorXd value(const Eigen::Vector3d&) const override {
        ++counts->values;
        return data.reshaped();
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        ++counts->derivatives;
        return Eigen::MatrixXd::Identity(data.size(), data.size());
    }
};
TEST(MaterialLocalEnergy3D, AccumulationReusesInputValuesAndTheActiveDerivative) {
    const CountingField<Eigen::Matrix3d> F{1.1 * Eigen::Matrix3d::Identity()}, Fp{0.95 * Eigen::Matrix3d::Identity()};
    const CountingField<Eigen::VectorXd> theta{Eigen::Vector2d(2, 3)};
    const auto eF = litepgo::MaterialLocalEnergyF3D<NeoHookean>(F, Fp, theta);
    const auto ep = litepgo::MaterialLocalEnergyFp3D<NeoHookean>(F, Fp, theta);
    const auto et = litepgo::MaterialLocalEnergyTheta3D<NeoHookean>(F, Fp, theta);
    for (const LocalEnergy3D* energy : {static_cast<const LocalEnergy3D*>(&eF), static_cast<const LocalEnergy3D*>(&ep),
                                        static_cast<const LocalEnergy3D*>(&et)}) {
        const Eigen::Vector3d xi(0.2, 0.3, 0.4);
        for (int request = 0; request < 4; ++request) {
            const bool want_g = request & 1, want_h = request & 2;
            Eigen::VectorXd g = Eigen::VectorXd::Constant(energy->num_dofs(), 2);
            Eigen::MatrixXd H = Eigen::MatrixXd::Constant(g.size(), g.size(), 3);
            const double weight = -0.4;
            const double dV = weight;
            const double expected = dV * energy->value(xi);
            Eigen::VectorXd expected_g = g;
            Eigen::MatrixXd expected_H = H;
            if (want_g)
                expected_g += dV * energy->gradient(xi);
            if (want_h)
                expected_H += dV * energy->hessian(xi);
            *F.counts = {};
            *Fp.counts = {};
            *theta.counts = {};
            EXPECT_NEAR(energy->accumulate(xi, weight, want_g ? &g : nullptr, want_h ? &H : nullptr), expected, 1e-13);
            EXPECT_TRUE(g.isApprox(expected_g, 1e-13));
            EXPECT_TRUE(H.isApprox(expected_H, 1e-13));
            EXPECT_EQ(F.counts->values, 1);
            EXPECT_EQ(Fp.counts->values, 1);
            EXPECT_EQ(theta.counts->values, 1);
            EXPECT_EQ(F.counts->derivatives + Fp.counts->derivatives + theta.counts->derivatives,
                      want_g || want_h ? 1 : 0);
        }
        Eigen::VectorXd g = Eigen::VectorXd::Constant(energy->num_dofs(), 7);
        const auto before = g;
        EXPECT_THROW(energy->accumulate(xi, std::numeric_limits<double>::infinity(), &g, nullptr),
                     std::invalid_argument);
        EXPECT_EQ(g, before);
    }
}
} // namespace

namespace {
struct BadActiveField final : Field3D {
    int num_components() const override {
        return 9;
    }
    int n = 9;
    Eigen::MatrixXd derivative = Eigen::MatrixXd::Zero(9, 8);
    int num_dofs() const override {
        return n;
    }
    Eigen::VectorXd value(const Eigen::Vector3d&) const override {
        return Eigen::Matrix3d::Identity().reshaped();
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        return derivative;
    }
};
TEST(MaterialLocalEnergy3D, RejectsInvalidActiveDerivativeShapesCountsAndNonFiniteEntries) {
    const ValueOnlyParameters parameters;
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const auto bind = [&](const BadActiveField& F) {
        return litepgo::MaterialLocalEnergyF3D<NeoHookean>(F, Fp, parameters);
    };
    BadActiveField bad;
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_THROW(bind(bad).gradient(xi), std::invalid_argument);
    bad.derivative = Eigen::MatrixXd::Zero(8, 9);
    EXPECT_THROW(bind(bad).hessian(xi), std::invalid_argument);
    bad.derivative = Eigen::MatrixXd::Identity(9, 9);
    bad.derivative(0, 0) = std::numeric_limits<double>::infinity();
    Eigen::VectorXd g = Eigen::VectorXd::Constant(9, 7);
    EXPECT_THROW(bind(bad).accumulate(xi, 1, &g, nullptr), std::invalid_argument);
    EXPECT_EQ(g, Eigen::VectorXd::Constant(9, 7));
    bad.n = 0;
    EXPECT_THROW(bind(bad), std::invalid_argument);
}
struct SpatialOnlyDisplacement final : Field3D {
    int num_components() const override {
        return 3;
    }
    int num_dofs() const override {
        return 0;
    }
    Eigen::VectorXd value(const Eigen::Vector3d& xi) const override {
        return 0.1 * xi;
    }
    Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d&) const override {
        return Eigen::Matrix3d::Identity() * 0.1;
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        throw std::logic_error("Read-only displacement");
    }
};
TEST(MaterialLocalEnergy3D, InactiveDeformationNeedsNoCoefficientDerivative) {
    const SpatialOnlyDisplacement u_field;
    const auto X_field = coordinates();
    const DeformationGradientField3D deformation(u_field, X_field);
    const ValueOnlyParameters parameters;
    const auto Fp = IdentityField3D(Eigen::Matrix3d::Identity().reshaped());
    const auto energy = litepgo::MaterialLocalEnergyFp3D<NeoHookean>(deformation, Fp, parameters);
    EXPECT_GT(energy.value({0.2, 0.3, 0.4}), 0);
    EXPECT_EQ(energy.hessian({0.2, 0.3, 0.4}).rows(), 9);
    EXPECT_THROW((litepgo::MaterialLocalEnergyF3D<NeoHookean>(deformation, Fp, parameters)), std::invalid_argument);
}
} // namespace

TEST(MaterialLocalEnergy3D, RejectsInvalidFAndFpSizesBeforeReshaping) {
    const ValueOnlyParameters parameters;
    const auto Fp = litepgo::IdentityField3D(Eigen::Matrix3d::Identity().reshaped());
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    for (int size : {1, 4, 8, 10}) {
        SCOPED_TRACE(size);
        const litepgo::IdentityField3D invalid(Eigen::VectorXd::Ones(size));
        const auto invalid_F = litepgo::MaterialLocalEnergyFp3D<litepgo::NeoHookean>(invalid, Fp, parameters);
        const auto invalid_Fp = litepgo::MaterialLocalEnergyFp3D<litepgo::NeoHookean>(Fp, invalid, parameters);
        for (const auto* energy : {&invalid_F, &invalid_Fp}) {
            EXPECT_THROW(energy->value(xi), std::invalid_argument);
            EXPECT_THROW(energy->gradient(xi), std::invalid_argument);
            EXPECT_THROW(energy->hessian(xi), std::invalid_argument);
        }
    }
}

namespace {
struct InvalidParameterGradients {
    static constexpr int num_basis_functions = 8;
    using Sample = litepgo::HexQ1Basis::Sample;
    static Sample evaluate(const Eigen::Vector3d& xi) {
        auto basis_sample = litepgo::HexQ1Basis::evaluate(xi);
        basis_sample.dN_dxi.setConstant(std::numeric_limits<double>::quiet_NaN());
        return basis_sample;
    }
};
TEST(MaterialLocalEnergy3D, InactiveInterpolatedParametersDoNotRequireSpatialDerivatives) {
    Eigen::Matrix<double, 8, 2> theta_local;
    theta_local.col(0).setConstant(2);
    theta_local.col(1).setConstant(3);
    const auto theta_field = litepgo::InterpolatedField3D<InvalidParameterGradients>(theta_local);
    const auto X_field = coordinates();
    const litepgo::InterpolatedField3D<litepgo::HexQ1Basis> u_field(Eigen::MatrixXd::Zero(8, 3));
    const litepgo::DeformationGradientField3D deformation(u_field, X_field);
    const litepgo::IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const auto energy = litepgo::MaterialLocalEnergyF3D<litepgo::NeoHookean>(deformation, Fp_field, theta_field);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(energy.num_dofs());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(energy.num_dofs(), energy.num_dofs());
    EXPECT_DOUBLE_EQ(energy.accumulate({0.2, 0.3, 0.4}, 1, &g, &H), 0);
    EXPECT_TRUE(g.isZero());
    EXPECT_GT(H.norm(), 0);
}
} // namespace
