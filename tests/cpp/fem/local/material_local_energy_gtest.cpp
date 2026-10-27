#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/deformation_gradient.hpp"
#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "quadratic_parameter_material.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace {
using namespace litepgo;
auto coordinates(double scale = 1) {
    Eigen::Matrix<double, 8, 3> X_local;
    for (int a = 0; a < 8; ++a)
        X_local.row(a) = scale * HexQ1Basis::reference_node(a).transpose();
    return InterpolatedField<HexQ1Basis>(X_local);
}
struct ValueOnlyParameters {
    Eigen::VectorXd data = Eigen::Vector2d(2, 3);
    Eigen::VectorXd value(const Eigen::Vector3d&) const {
        return data;
    }
};
struct ValueOnlyF {
    Eigen::Matrix3d value(const Eigen::Vector3d&) const {
        return Eigen::Matrix3d::Identity();
    }
};
TEST(MaterialLocalEnergy, DirectFieldsBindCorrectDerivativesAndOwnSnapshots) {
    Eigen::Matrix3d F = (Eigen::Matrix3d() << 1.2, 0.1, -0.03, 0.04, 0.9, 0.02, -0.01, 0.04, 1.1).finished();
    const Eigen::Matrix3d Fp = (Eigen::Matrix3d() << 0.95, -0.02, 0.05, 0.01, 1.1, -0.04, 0.03, 0.01, 1.05).finished();
    const auto X_field = coordinates(2);
    const IdentityField F_field(F.reshaped()), Fp_field(Fp.reshaped());
    const IdentityField theta_field(Eigen::Vector2d(2, 3));
    const auto energy =
        make_material_local_energy<NeoHookean, MaterialVariable::F>(F_field, Fp_field, theta_field, X_field);
    const auto plastic =
        make_material_local_energy<NeoHookean, MaterialVariable::Fp>(F_field, Fp_field, theta_field, X_field);
    const auto parameters =
        make_material_local_energy<NeoHookean, MaterialVariable::Theta>(F_field, Fp_field, theta_field, X_field);
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
    EXPECT_NEAR(integrate_volume<GaussLegendre3D<2>>(energy), 8 * W, 1e-13);
    F.setZero();
    EXPECT_DOUBLE_EQ(energy.value(xi), W);
    const auto detached = [&] {
        return make_material_local_energy<NeoHookean, MaterialVariable::F>(IdentityField(inputs.F.reshaped()), Fp_field,
                                                                           theta_field, coordinates());
    }();
    EXPECT_DOUBLE_EQ(detached.value(xi), W);
}
TEST(MaterialLocalEnergy, InactiveInputsNeedOnlyValuesAndActiveCapabilitiesAreCheckedAtBinding) {
    const IdentityField Fp(Eigen::Matrix3d::Identity().reshaped());
    const auto energy = make_material_local_energy<NeoHookean, MaterialVariable::Fp>(
        ValueOnlyF{}, Fp, ValueOnlyParameters{}, coordinates());
    EXPECT_DOUBLE_EQ(energy.value({0.2, 0.3, 0.4}), 0);
    EXPECT_EQ(energy.hessian({0.2, 0.3, 0.4}).rows(), 9);
    EXPECT_THROW((make_material_local_energy<NeoHookean, MaterialVariable::F>(ValueOnlyF{}, Fp, ValueOnlyParameters{},
                                                                              coordinates())),
                 std::invalid_argument);
    EXPECT_THROW((make_material_local_energy<NeoHookean, MaterialVariable::Theta>(
                     ValueOnlyF{}, Fp, ValueOnlyParameters{}, coordinates())),
                 std::invalid_argument);
    ValueOnlyParameters bad;
    bad.data.resize(0);
    const auto empty =
        make_material_local_energy<NeoHookean, MaterialVariable::Fp>(ValueOnlyF{}, Fp, bad, coordinates());
    EXPECT_THROW(empty.value({0.2, 0.3, 0.4}), std::invalid_argument);
    bad.data = Eigen::Vector2d(2, std::numeric_limits<double>::infinity());
    const auto nonfinite =
        make_material_local_energy<NeoHookean, MaterialVariable::Fp>(ValueOnlyF{}, Fp, bad, coordinates());
    EXPECT_THROW(nonfinite.hessian({0.2, 0.3, 0.4}), std::invalid_argument);
}
struct MappedParameters {
    Eigen::Vector4d coeffs;
    Eigen::Matrix<double, 3, 4> derivative =
        (Eigen::Matrix<double, 3, 4>() << 1, 0.2, 0, -0.3, 0, 2, 0.4, 0, -0.1, 0, 1.5, 0.6).finished();
    struct Sample {
        const Eigen::Vector3d value;
        const Eigen::Matrix<double, 3, 4> derivative;
        Eigen::MatrixXd dvalue_dcoeffs() const {
            return derivative;
        }
    };
    int num_dofs() const {
        return 4;
    }
    Sample evaluate(const Eigen::Vector3d&) const {
        return {derivative * coeffs, derivative};
    }
};
TEST(MaterialLocalEnergy, AnotherMaterialUsesOnlyItsActiveDerivativesAndOwnParameterCount) {
    const IdentityField F(Eigen::Matrix3d::Identity().reshaped());
    const auto bind = [&](const Eigen::Vector4d& coeffs) {
        return make_material_local_energy<QuadraticParameterMaterial<3>, MaterialVariable::Theta>(
            F, F, MappedParameters{coeffs}, coordinates());
    };
    const Eigen::Vector4d x(0.2, -0.3, 0.5, 0.8);
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    const auto energy = bind(x);
    const auto g = energy.gradient(xi);
    const auto H = energy.hessian(xi);
    for (int j = 0; j < 4; ++j) {
        auto plus = x, minus = x;
        plus[j] += 1e-4;
        minus[j] -= 1e-4;
        EXPECT_NEAR(g[j], (bind(plus).value(xi) - bind(minus).value(xi)) / 2e-4, 1e-10);
        EXPECT_TRUE(H.col(j).isApprox((bind(plus).gradient(xi) - bind(minus).gradient(xi)) / 2e-4, 1e-10));
    }
}
struct Counts {
    int values = 0, derivatives = 0;
};
template <class Value> struct CountingField {
    Value data;
    std::shared_ptr<Counts> counts = std::make_shared<Counts>();
    struct Sample {
        const Value value;
        std::shared_ptr<Counts> counts;
        Eigen::MatrixXd dvalue_dcoeffs() const {
            ++counts->derivatives;
            return Eigen::MatrixXd::Identity(value.size(), value.size());
        }
    };
    int num_dofs() const {
        return static_cast<int>(data.size());
    }
    Sample evaluate(const Eigen::Vector3d&) const {
        ++counts->values;
        return {data, counts};
    }
};
TEST(MaterialLocalEnergy, AccumulationReusesInputValuesAndTheActiveDerivative) {
    const CountingField<Eigen::Matrix3d> F{1.1 * Eigen::Matrix3d::Identity()}, Fp{0.95 * Eigen::Matrix3d::Identity()};
    const CountingField<Eigen::VectorXd> theta{Eigen::Vector2d(2, 3)};
    const auto eF = make_material_local_energy<NeoHookean, MaterialVariable::F>(F, Fp, theta, coordinates(2));
    const auto ep = make_material_local_energy<NeoHookean, MaterialVariable::Fp>(F, Fp, theta, coordinates(2));
    const auto et = make_material_local_energy<NeoHookean, MaterialVariable::Theta>(F, Fp, theta, coordinates(2));
    for (const LocalEnergy* energy : {static_cast<const LocalEnergy*>(&eF), static_cast<const LocalEnergy*>(&ep),
                                      static_cast<const LocalEnergy*>(&et)}) {
        const Eigen::Vector3d xi(0.2, 0.3, 0.4);
        for (int request = 0; request < 4; ++request) {
            const bool want_g = request & 1, want_h = request & 2;
            Eigen::VectorXd g = Eigen::VectorXd::Constant(energy->num_dofs(), 2);
            Eigen::MatrixXd H = Eigen::MatrixXd::Constant(g.size(), g.size(), 3);
            const double weight = -0.4;
            const double dV = 8 * weight;
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
struct CountingCoordinates {
    InterpolatedField<HexQ1Basis> field = coordinates(2);
    std::shared_ptr<int> calls = std::make_shared<int>(0);
    int num_components() const {
        return 3;
    }
    auto evaluate(const Eigen::Vector3d& xi) const {
        ++*calls;
        return field.evaluate(xi);
    }
};
TEST(MaterialLocalEnergy, DeformationAndVolumeMeasureShareOneCoordinateSample) {
    const CountingCoordinates X_field;
    const auto deformation = DeformationGradient(InterpolatedField<HexQ1Basis>(Eigen::MatrixXd::Zero(8, 3)), X_field);
    const auto energy = make_material_local_energy<NeoHookean, MaterialVariable::F>(
        deformation, IdentityField(Eigen::Matrix3d::Identity().reshaped()), IdentityField(Eigen::Vector2d(2, 3)));
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    EXPECT_DOUBLE_EQ(integrate_volume<GaussLegendre3D<2>>(energy, &g, &H), 0);
    EXPECT_EQ(*X_field.calls, GaussLegendre3D<2>::num_points);
}
} // namespace

namespace {
struct BadActiveField {
    int n = 9;
    Eigen::MatrixXd derivative = Eigen::MatrixXd::Zero(9, 8);
    struct Sample {
        const Eigen::Matrix3d value = Eigen::Matrix3d::Identity();
        const Eigen::MatrixXd derivative;
        Eigen::MatrixXd dvalue_dcoeffs() const {
            return derivative;
        }
    };
    int num_dofs() const {
        return n;
    }
    Sample evaluate(const Eigen::Vector3d&) const {
        return {Eigen::Matrix3d::Identity(), derivative};
    }
};
TEST(MaterialLocalEnergy, RejectsInvalidActiveDerivativeShapesCountsAndNonFiniteEntries) {
    const IdentityField Fp(Eigen::Matrix3d::Identity().reshaped());
    const auto bind = [&](const BadActiveField& F) {
        return make_material_local_energy<NeoHookean, MaterialVariable::F>(F, Fp, ValueOnlyParameters{}, coordinates());
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
struct SpatialOnlyDisplacement {
    struct Sample {
        const Eigen::Vector3d value;
        const Eigen::Matrix3d dvalue_dxi = Eigen::Matrix3d::Identity() * 0.1;
    };
    int num_components() const {
        return 3;
    }
    Sample evaluate(const Eigen::Vector3d& xi) const {
        return {0.1 * xi};
    }
};
TEST(MaterialLocalEnergy, InactiveDeformationNeedsNoCoefficientDerivativeAndRejectsIndependentGeometry) {
    const auto deformation = DeformationGradient(SpatialOnlyDisplacement{}, coordinates());
    const auto Fp = IdentityField(Eigen::Matrix3d::Identity().reshaped());
    const auto energy =
        make_material_local_energy<NeoHookean, MaterialVariable::Fp>(deformation, Fp, ValueOnlyParameters{});
    EXPECT_GT(energy.value({0.2, 0.3, 0.4}), 0);
    EXPECT_EQ(energy.hessian({0.2, 0.3, 0.4}).rows(), 9);
    EXPECT_THROW((make_material_local_energy<NeoHookean, MaterialVariable::F>(deformation, Fp, ValueOnlyParameters{})),
                 std::invalid_argument);
}
template <class Source>
concept CanBindIndependentGeometry = requires(const Source& source) {
    make_material_local_energy<NeoHookean, MaterialVariable::F>(
        source, IdentityField(Eigen::Matrix3d::Identity().reshaped()), ValueOnlyParameters{}, coordinates());
};
using CoordinateField = InterpolatedField<HexQ1Basis>;
static_assert(!CanBindIndependentGeometry<DeformationGradient<CoordinateField, CoordinateField>>);
static_assert(CanBindIndependentGeometry<IdentityField>);
} // namespace

TEST(MaterialLocalEnergy, RejectsInvalidFAndFpSizesBeforeReshaping) {
    const auto Fp = litepgo::IdentityField(Eigen::Matrix3d::Identity().reshaped());
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    for (int size : {1, 4, 8, 10}) {
        SCOPED_TRACE(size);
        const litepgo::IdentityField invalid(Eigen::VectorXd::Ones(size));
        const auto invalid_F = litepgo::make_material_local_energy<litepgo::NeoHookean, litepgo::MaterialVariable::Fp>(
            invalid, Fp, ValueOnlyParameters{}, coordinates());
        const auto invalid_Fp = litepgo::make_material_local_energy<litepgo::NeoHookean, litepgo::MaterialVariable::Fp>(
            Fp, invalid, ValueOnlyParameters{}, coordinates());
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
TEST(MaterialLocalEnergy, InactiveInterpolatedParametersDoNotRequireSpatialDerivatives) {
    Eigen::Matrix<double, 8, 2> theta_local;
    theta_local.col(0).setConstant(2);
    theta_local.col(1).setConstant(3);
    const auto theta_field = litepgo::InterpolatedField<InvalidParameterGradients>(theta_local);
    const auto deformation = litepgo::DeformationGradient(
        litepgo::InterpolatedField<litepgo::HexQ1Basis>(Eigen::MatrixXd::Zero(8, 3)), coordinates());
    const auto energy = litepgo::make_material_local_energy<litepgo::NeoHookean, litepgo::MaterialVariable::F>(
        deformation, litepgo::IdentityField(Eigen::Matrix3d::Identity().reshaped()), theta_field);
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    EXPECT_DOUBLE_EQ(litepgo::integrate_volume<litepgo::GaussLegendre3D<2>>(energy, &g, &H), 0);
    EXPECT_TRUE(g.isZero());
    EXPECT_GT(H.norm(), 0);
}
} // namespace
