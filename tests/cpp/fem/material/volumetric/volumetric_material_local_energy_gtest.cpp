#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace litepgo::fem::test {

namespace {
auto coordinates(double scale = 1) {
    Eigen::Matrix<double, 8, 3> X_local;
    for (int a = 0; a < 8; ++a)
        X_local.row(a) = scale * HexQ1Basis::reference_node(a).transpose();
    return InterpolatedField3D<HexQ1Basis>(X_local);
}

struct QuadraticDeformationMaterial {
    static double eval_W(const VolumetricMaterialInputs& inputs) {
        return 0.5 * inputs.F.squaredNorm();
    }
    static Eigen::Matrix3d eval_dW_dF(const VolumetricMaterialInputs& inputs) {
        return inputs.F;
    }
    static Eigen::Matrix<double, 9, 9> eval_d2W_dF2(const VolumetricMaterialInputs&) {
        return Eigen::Matrix<double, 9, 9>::Identity();
    }
};

TEST(VolumetricMaterialLocalEnergyF3D, OverflowingDensityWeightAndAccumulationsThrowDomainError) {
    IdentityField3D F(Eigen::VectorXd::Constant(9, 1e155));
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
    const VolumetricMaterialLocalEnergyF3D<QuadraticDeformationMaterial> energy(F, Fp, theta);
    const Eigen::Vector3d xi = Eigen::Vector3d::Zero();
    EXPECT_THROW(energy.eval_value(xi), std::domain_error);
    EXPECT_THROW(energy.accumulate(xi, 1, nullptr, nullptr), std::domain_error);
    F = IdentityField3D(Eigen::VectorXd::Constant(9, 1e150));
    ASSERT_TRUE(std::isfinite(energy.eval_value(xi)));
    EXPECT_THROW(energy.accumulate(xi, 1e100, nullptr, nullptr), std::domain_error);
    F = IdentityField3D(Eigen::VectorXd::Ones(9));
    const double max = std::numeric_limits<double>::max();
    Eigen::VectorXd g = Eigen::VectorXd::Constant(9, max);
    Eigen::MatrixXd H = Eigen::MatrixXd::Constant(9, 9, max);
    EXPECT_THROW(energy.accumulate(xi, max, &g, nullptr), std::domain_error);
    EXPECT_THROW(energy.accumulate(xi, max, nullptr, &H), std::domain_error);
}
struct ValueOnlyField : Field3D {
    int num_dofs() const override {
        return 0;
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        throw std::logic_error("Read-only field");
    }
};
struct ValueOnlyParameters final : ValueOnlyField {
    Eigen::VectorXd data = Eigen::Vector2d(2, 3);
    int num_components() const override {
        return static_cast<int>(data.size());
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        return data;
    }
};
struct ValueOnlyF final : ValueOnlyField {
    int num_components() const override {
        return 9;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        return Eigen::Matrix3d::Identity().reshaped();
    }
};
TEST(VolumetricMaterialLocalEnergyF3D, DirectFieldsBindCorrectDerivativesAndBorrowNamedFields) {
    Eigen::Matrix3d F = (Eigen::Matrix3d() << 1.2, 0.1, -0.03, 0.04, 0.9, 0.02, -0.01, 0.04, 1.1).finished();
    const Eigen::Matrix3d Fp = (Eigen::Matrix3d() << 0.95, -0.02, 0.05, 0.01, 1.1, -0.04, 0.03, 0.01, 1.05).finished();
    IdentityField3D F_field(F.reshaped());
    const IdentityField3D Fp_field(Fp.reshaped());
    const IdentityField3D theta_field(Eigen::Vector2d(2, 3));
    const auto energy = litepgo::fem::VolumetricMaterialLocalEnergyF3D<NeoHookean>(F_field, Fp_field, theta_field);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const VolumetricMaterialInputs inputs{F, Fp, Eigen::Vector2d(2, 3)};
    const double W = NeoHookean::eval_W(inputs);
    EXPECT_DOUBLE_EQ(energy.eval_value(xi), W);
    EXPECT_TRUE(energy.eval_gradient(xi).isApprox(NeoHookean::eval_dW_dF(inputs).reshaped(), 1e-14));
    EXPECT_EQ(energy.eval_hessian(xi), NeoHookean::eval_d2W_dF2(inputs));
    EXPECT_NEAR(energy.accumulate(xi, 8, nullptr, nullptr), 8 * W, 1e-13);
    F.setZero();
    EXPECT_DOUBLE_EQ(energy.eval_value(xi), W);
    F = 1.01 * inputs.F;
    F_field = IdentityField3D(F.reshaped());
    const auto copy = energy;
    EXPECT_DOUBLE_EQ(copy.eval_value(xi), NeoHookean::eval_W({F, Fp, inputs.theta}));
}
TEST(VolumetricMaterialLocalEnergyF3D, InactiveInputsNeedOnlyValuesAndActiveCapabilitiesAreCheckedAtBinding) {
    const ValueOnlyF readonly_F;
    const ValueOnlyParameters readonly_theta;
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const auto energy = litepgo::fem::VolumetricMaterialLocalEnergyF3D<NeoHookean>(Fp, readonly_F, readonly_theta);
    EXPECT_DOUBLE_EQ(energy.eval_value({0.2, 0.3, 0.4}), 0);
    EXPECT_EQ(energy.eval_hessian({0.2, 0.3, 0.4}).rows(), 9);
    EXPECT_THROW((litepgo::fem::VolumetricMaterialLocalEnergyF3D<NeoHookean>(readonly_F, Fp, readonly_theta)),
                 std::invalid_argument);
    ValueOnlyParameters bad;
    bad.data.resize(0);
    const auto empty = litepgo::fem::VolumetricMaterialLocalEnergyF3D<NeoHookean>(Fp, readonly_F, bad);
    EXPECT_THROW(empty.eval_value({0.2, 0.3, 0.4}), std::invalid_argument);
    bad.data = Eigen::Vector2d(2, std::numeric_limits<double>::infinity());
    const auto nonfinite = litepgo::fem::VolumetricMaterialLocalEnergyF3D<NeoHookean>(Fp, readonly_F, bad);
    EXPECT_THROW(nonfinite.eval_hessian({0.2, 0.3, 0.4}), std::invalid_argument);
}
struct MappedDeformation final : Field3D {
    explicit MappedDeformation(const Eigen::Vector4d& data) : coeffs(data) {}
    int num_components() const override {
        return 9;
    }
    Eigen::Vector4d coeffs;
    Eigen::Matrix<double, 9, 4> derivative = Eigen::Matrix<double, 9, 4>::NullaryExpr(
        [](Eigen::Index row, Eigen::Index col) { return std::sin(static_cast<double>(1 + row + 2 * col)); });
    int num_dofs() const override {
        return 4;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        return derivative * coeffs;
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        return derivative;
    }
};
TEST(VolumetricMaterialLocalEnergyF3D, AnotherMaterialSupportsRectangularDeformationPullback) {
    const IdentityField3D F(Eigen::Matrix3d::Identity().reshaped());
    const Eigen::Vector4d x(0.2, -0.3, 0.5, 0.8);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const MappedDeformation deformation(x);
    const VolumetricMaterialLocalEnergyF3D<QuadraticDeformationMaterial> energy(deformation, F, F);
    const auto g = energy.eval_gradient(xi);
    const auto H = energy.eval_hessian(xi);
    for (int j = 0; j < 4; ++j) {
        auto plus = x, minus = x;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        const MappedDeformation plus_deformation(plus), minus_deformation(minus);
        const VolumetricMaterialLocalEnergyF3D<QuadraticDeformationMaterial> plus_energy(plus_deformation, F, F);
        const VolumetricMaterialLocalEnergyF3D<QuadraticDeformationMaterial> minus_energy(minus_deformation, F, F);
        EXPECT_NEAR(g[j], (plus_energy.eval_value(xi) - minus_energy.eval_value(xi)) / 2e-4, 1e-10);
        EXPECT_TRUE(H.col(j).isApprox((plus_energy.eval_gradient(xi) - minus_energy.eval_gradient(xi)) / 2e-4, 1e-10));
    }
}

template <class Energy> void check_temporary_inputs_rejected() {
    static_assert(!std::is_constructible_v<Energy, IdentityField3D&&, const IdentityField3D&, const IdentityField3D&>);
    static_assert(
        !std::is_constructible_v<Energy, const IdentityField3D&, const IdentityField3D&&, const IdentityField3D&>);
    static_assert(!std::is_constructible_v<Energy, const IdentityField3D&, const IdentityField3D&, IdentityField3D&&>);
}

TEST(VolumetricMaterialLocalEnergyF3D, RejectsTemporaryFields) {
    static_assert(!std::is_abstract_v<VolumetricMaterialLocalEnergyF3D<NeoHookean>>);
    check_temporary_inputs_rejected<VolumetricMaterialLocalEnergyF3D<NeoHookean>>();
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
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        ++counts->values;
        return data.reshaped();
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        ++counts->derivatives;
        return Eigen::MatrixXd::Identity(data.size(), data.size());
    }
};
TEST(VolumetricMaterialLocalEnergyF3D, AccumulationReusesInputValuesAndTheActiveDerivative) {
    const CountingField<Eigen::Matrix3d> F{1.1 * Eigen::Matrix3d::Identity()}, Fp{0.95 * Eigen::Matrix3d::Identity()};
    const CountingField<Eigen::VectorXd> theta{Eigen::Vector2d(2, 3)};
    const auto energy = litepgo::fem::VolumetricMaterialLocalEnergyF3D<NeoHookean>(F, Fp, theta);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    for (int request = 0; request < 4; ++request) {
        const bool want_g = request & 1, want_h = request & 2;
        Eigen::VectorXd g = Eigen::VectorXd::Constant(energy.num_dofs(), 2);
        Eigen::MatrixXd H = Eigen::MatrixXd::Constant(g.size(), g.size(), 3);
        const double weight = -0.4;
        const double dV = weight;
        const double expected = dV * energy.eval_value(xi);
        Eigen::VectorXd expected_g = g;
        Eigen::MatrixXd expected_H = H;
        if (want_g)
            expected_g += dV * energy.eval_gradient(xi);
        if (want_h)
            expected_H += dV * energy.eval_hessian(xi);
        *F.counts = {};
        *Fp.counts = {};
        *theta.counts = {};
        EXPECT_NEAR(energy.accumulate(xi, weight, want_g ? &g : nullptr, want_h ? &H : nullptr), expected, 1e-13);
        EXPECT_TRUE(g.isApprox(expected_g, 1e-13));
        EXPECT_TRUE(H.isApprox(expected_H, 1e-13));
        EXPECT_EQ(F.counts->values, 1);
        EXPECT_EQ(Fp.counts->values, 1);
        EXPECT_EQ(theta.counts->values, 1);
        EXPECT_EQ(F.counts->derivatives + Fp.counts->derivatives + theta.counts->derivatives, want_g || want_h ? 1 : 0);
    }
    Eigen::VectorXd g = Eigen::VectorXd::Constant(energy.num_dofs(), 7);
    const auto before = g;
    EXPECT_THROW(energy.accumulate(xi, std::numeric_limits<double>::infinity(), &g, nullptr), std::invalid_argument);
    EXPECT_EQ(g, before);
}
TEST(VolumetricMaterialLocalEnergyF3D, FeasibleStepMapsOnlyTheActiveCoefficientDirection) {
    Eigen::Matrix3d A;
    A << 1.2, 0.2, 0, 0, 0.9, 0.1, 0.05, 0, 1.1;
    const IdentityField3D F(A.reshaped()), Fp(A.reshaped());
    const ValueOnlyParameters parameters;
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> total(F, Fp, parameters);
    const Eigen::Matrix3d dA = A * Eigen::Vector3d(-4, -4, 0).asDiagonal();
    const Eigen::VectorXd direction = dA.reshaped();
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const double expected =
        NeoHookean::eval_feasible_step({A, A, parameters.data}, {dA, Eigen::Matrix3d::Zero(), Eigen::Vector2d::Zero()});
    EXPECT_EQ(total.eval_feasible_step(xi, direction), expected);
    EXPECT_THROW(total.eval_feasible_step(xi, Eigen::VectorXd::Zero(8)), std::invalid_argument);
    Eigen::VectorXd bad = direction;
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(total.eval_feasible_step(xi, bad), std::invalid_argument);
    EXPECT_THROW(
        total.eval_feasible_step(Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity()), direction),
        std::invalid_argument);
}

TEST(VolumetricMaterialLocalEnergyF3D, FeasibleStepUsesTheDeformationFieldAffineMapWithoutAnIdentityIncrement) {
    const auto X = coordinates(2);
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(8, 3), du(8, 3);
    const Eigen::Matrix3d dF = Eigen::Vector3d(-4, -4, 0).asDiagonal();
    for (int v = 0; v < 8; ++v)
        du.row(v) = (dF * (2 * HexQ1Basis::reference_node(v))).transpose();
    const InterpolatedField3D<HexQ1Basis> displacement(u);
    const DeformationGradientField3D deformation(displacement, X);
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped()), theta(Eigen::Vector2d(2, 3));
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> local(deformation, Fp, theta);
    const Eigen::VectorXd direction = du.reshaped<Eigen::RowMajor>();
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const double alpha = local.eval_feasible_step(xi, direction);
    EXPECT_NEAR(alpha, 0.225, 1e-6);
    for (int i = 0; i <= 20; ++i) {
        const InterpolatedField3D<HexQ1Basis> trial_u((alpha * i / 20) * du);
        const DeformationGradientField3D trial_F(trial_u, X);
        const VolumetricMaterialLocalEnergyF3D<NeoHookean> trial(trial_F, Fp, theta);
        EXPECT_TRUE(std::isfinite(trial.eval_value(xi)));
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
    Eigen::VectorXd eval_value(const Eigen::Vector3d&) const override {
        return Eigen::Matrix3d::Identity().reshaped();
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        return derivative;
    }
};
TEST(VolumetricMaterialLocalEnergyF3D, RejectsInvalidActiveDerivativeShapesCountsAndNonFiniteEntries) {
    const ValueOnlyParameters parameters;
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const auto bind = [&](const BadActiveField& F) {
        return litepgo::fem::VolumetricMaterialLocalEnergyF3D<NeoHookean>(F, Fp, parameters);
    };
    BadActiveField bad;
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_THROW(bind(bad).eval_gradient(xi), std::invalid_argument);
    bad.derivative = Eigen::MatrixXd::Zero(8, 9);
    EXPECT_THROW(bind(bad).eval_hessian(xi), std::invalid_argument);
    bad.derivative = Eigen::MatrixXd::Identity(9, 9);
    bad.derivative(0, 0) = std::numeric_limits<double>::infinity();
    Eigen::VectorXd g = Eigen::VectorXd::Constant(9, 7);
    EXPECT_THROW(bind(bad).accumulate(xi, 1, &g, nullptr), std::invalid_argument);
    EXPECT_EQ(g, Eigen::VectorXd::Constant(9, 7));
    bad.n = 0;
    EXPECT_THROW(bind(bad), std::invalid_argument);
}

TEST(VolumetricMaterialLocalEnergyF3D, OverflowingMappedFeasibleDirectionThrowsDomainError) {
    BadActiveField F;
    F.derivative = 1e200 * Eigen::MatrixXd::Identity(9, 9);
    const IdentityField3D Fp(Eigen::Matrix3d::Identity().reshaped());
    const ValueOnlyParameters parameters;
    const VolumetricMaterialLocalEnergyF3D<NeoHookean> local(F, Fp, parameters);
    const Eigen::VectorXd direction = Eigen::VectorXd::Constant(9, 1e200);
    EXPECT_THROW(local.eval_feasible_step(Eigen::Vector3d::Zero(), direction), std::domain_error);
}

} // namespace

TEST(VolumetricMaterialLocalEnergyF3D, RejectsInvalidFAndFpSizesBeforeReshaping) {
    const ValueOnlyParameters parameters;
    const auto Fp = litepgo::fem::IdentityField3D(Eigen::Matrix3d::Identity().reshaped());
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    for (int size : {1, 4, 8, 10}) {
        SCOPED_TRACE(size);
        const litepgo::fem::IdentityField3D invalid(Eigen::VectorXd::Ones(size));
        const auto invalid_F =
            litepgo::fem::VolumetricMaterialLocalEnergyF3D<litepgo::fem::NeoHookean>(invalid, Fp, parameters);
        const auto invalid_Fp =
            litepgo::fem::VolumetricMaterialLocalEnergyF3D<litepgo::fem::NeoHookean>(Fp, invalid, parameters);
        for (const auto* energy : {&invalid_F, &invalid_Fp}) {
            EXPECT_THROW(energy->eval_value(xi), std::invalid_argument);
            EXPECT_THROW(energy->eval_gradient(xi), std::invalid_argument);
            EXPECT_THROW(energy->eval_hessian(xi), std::invalid_argument);
        }
    }
}

namespace {
struct InvalidParameterGradients {
    static constexpr int num_basis_functions = 8;
    static Eigen::Matrix<double, 8, 1> eval_N(const Eigen::Vector3d& xi) {
        return HexQ1Basis::eval_N(xi);
    }
    static Eigen::Matrix<double, 8, 3> eval_dN_dxi(const Eigen::Vector3d&) {
        return Eigen::Matrix<double, 8, 3>::Constant(std::numeric_limits<double>::quiet_NaN());
    }
    static Eigen::Matrix<double, 8, 6> eval_d2N_dxi2(const Eigen::Vector3d& xi) {
        return HexQ1Basis::eval_d2N_dxi2(xi);
    }
};
TEST(VolumetricMaterialLocalEnergyF3D, InactiveInterpolatedParametersDoNotRequireSpatialDerivatives) {
    Eigen::Matrix<double, 8, 2> theta_local;
    theta_local.col(0).setConstant(2);
    theta_local.col(1).setConstant(3);
    const auto theta_field = litepgo::fem::InterpolatedField3D<InvalidParameterGradients>(theta_local);
    const auto X_field = coordinates();
    const litepgo::fem::InterpolatedField3D<litepgo::fem::HexQ1Basis> u_field(Eigen::MatrixXd::Zero(8, 3));
    const litepgo::fem::DeformationGradientField3D deformation(u_field, X_field);
    const litepgo::fem::IdentityField3D Fp_field(Eigen::Matrix3d::Identity().reshaped());
    const auto energy =
        litepgo::fem::VolumetricMaterialLocalEnergyF3D<litepgo::fem::NeoHookean>(deformation, Fp_field, theta_field);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(energy.num_dofs());
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(energy.num_dofs(), energy.num_dofs());
    EXPECT_DOUBLE_EQ(energy.accumulate({0.2, 0.3, 0.4}, 1, &g, &H), 0);
    EXPECT_TRUE(g.isZero());
    EXPECT_GT(H.norm(), 0);
}
} // namespace

} // namespace litepgo::fem::test
