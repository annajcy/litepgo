#include "litepgo/fem/constant_field.hpp"
#include "litepgo/fem/deformation_gradient_field.hpp"
#include "litepgo/fem/element_field.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/local_energy.hpp"
#include "litepgo/fem/material_fields.hpp"
#include "litepgo/fem/neo_hookean_local_energy.hpp"

#include <gtest/gtest.h>

namespace {
using namespace litepgo;
template <class Fields>
concept CanBindAll = requires {
    typename NeoHookeanLocalEnergyF<Fields>;
    typename NeoHookeanLocalEnergyFp<Fields>;
    typename NeoHookeanLocalEnergyTheta<Fields>;
};
static_assert(CanBindAll<MaterialFields<ElementField<Eigen::Matrix3d>, ElementField<Eigen::Matrix3d>,
                                        ConstantField<Eigen::Vector2d>>>);
static_assert(!CanBindAll<MaterialFields<ElementField<Eigen::Matrix3d>, ElementField<Eigen::Matrix3d>,
                                         ConstantField<Eigen::Vector3d>>>);

TEST(NeoHookeanLocalEnergy, DirectFieldsBindCorrectDerivativesAndOwnSnapshots) {
    Eigen::Matrix<double, 8, 3> rest;
    for (int i = 0; i < 8; ++i)
        rest.row(i) = HexQ1Basis::reference_node(i).transpose();
    const VolumeReferenceGeometry<HexQ1Basis> geometry(rest);
    const auto point = geometry.evaluate({0.2, 0.3, 0.4});

    Eigen::Matrix3d F, Fp;
    F << 1.2, 0.1, -0.03, 0.04, 0.9, 0.02, -0.01, 0.04, 1.1;
    Fp << 0.95, -0.02, 0.05, 0.01, 1.1, -0.04, 0.03, 0.01, 1.05;
    MaterialFields fields{ElementField<Eigen::Matrix3d>(F), ElementField<Eigen::Matrix3d>(Fp),
                          ConstantField<Eigen::Vector2d>(Eigen::Vector2d(2, 3))};
    const NeoHookeanLocalEnergyF deformation(fields);
    const NeoHookeanLocalEnergyFp plastic(fields);
    const NeoHookeanLocalEnergyTheta parameters(fields);
    static_assert(std::same_as<std::remove_cvref_t<decltype(deformation)>, NeoHookeanLocalEnergyF<decltype(fields)>>);
    static_assert(std::same_as<std::remove_cvref_t<decltype(plastic)>, NeoHookeanLocalEnergyFp<decltype(fields)>>);
    static_assert(
        std::same_as<std::remove_cvref_t<decltype(parameters)>, NeoHookeanLocalEnergyTheta<decltype(fields)>>);
    static_assert(decltype(deformation)::num_dofs == 9);
    static_assert(LocalValue<decltype(deformation), VolumeReferenceGeometry<HexQ1Basis>::PointData>);
    static_assert(LocalValue<decltype(deformation), double>);
    static_assert(LocalFirstOrderEnergy<decltype(plastic), double>);
    static_assert(LocalSecondOrderEnergy<decltype(parameters), Eigen::Vector3d>);
    static_assert(
        LocalValue<decltype(deformation), VolumeReferenceGeometry<TensorProductBasis3D<LagrangeBasis>>::PointData>);

    static_assert(decltype(plastic)::num_dofs == 9);
    static_assert(decltype(parameters)::num_dofs == 2);
    static_assert(!std::is_constructible_v<decltype(deformation), decltype(fields), NeoHookeanEnergyFp>);
    const MaterialFields nodal_fields{DeformationGradientField<HexQ1Basis>(Eigen::Matrix<double, 8, 3>::Zero()),
                                      fields.Fp, fields.theta};
    const NeoHookeanLocalEnergyF nodal_deformation(nodal_fields);
    const NeoHookeanLocalEnergyFp nodal_plastic(nodal_fields);
    const NeoHookeanLocalEnergyTheta nodal_parameters(nodal_fields);
    static_assert(LocalValue<decltype(nodal_deformation), VolumeReferenceGeometry<HexQ1Basis>::PointData>);
    static_assert(LocalFirstOrderEnergy<decltype(nodal_plastic), VolumeReferenceGeometry<HexQ1Basis>::PointData>);
    static_assert(LocalSecondOrderEnergy<decltype(nodal_parameters), VolumeReferenceGeometry<HexQ1Basis>::PointData>);
    static_assert(!LocalValue<decltype(nodal_deformation), double>);
    static_assert(!LocalFirstOrderEnergy<decltype(nodal_plastic), double>);
    static_assert(!LocalSecondOrderEnergy<decltype(nodal_parameters), Eigen::Vector3d>);
    static_assert(!LocalValue<decltype(nodal_deformation),
                              VolumeReferenceGeometry<TensorProductBasis3D<LagrangeBasis>>::PointData>);
    const auto inputs = fields.evaluate(point);
    const double expected = NeoHookean{}.value(inputs);
    EXPECT_DOUBLE_EQ(deformation.value(point), expected);
    EXPECT_DOUBLE_EQ(plastic.value(point), expected);
    EXPECT_DOUBLE_EQ(parameters.value(point), expected);
    EXPECT_DOUBLE_EQ(deformation.value(0.3), expected);
    EXPECT_EQ(plastic.gradient(0.7), plastic.gradient(point));
    EXPECT_EQ(parameters.hessian(Eigen::Vector3d::Zero()), parameters.hessian(point));
    const auto stress = NeoHookeanEnergyF{}.gradient(inputs);
    const auto plastic_stress = NeoHookeanEnergyFp{}.gradient(inputs);
    EXPECT_TRUE(
        deformation.gradient(point).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(stress.data()), 1e-13));
    EXPECT_TRUE(
        plastic.gradient(point).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(plastic_stress.data()), 1e-13));
    EXPECT_EQ(parameters.gradient(point), NeoHookeanEnergyTheta{}.gradient(inputs));
    EXPECT_EQ(deformation.hessian(point), NeoHookeanEnergyF{}.hessian(inputs));
    EXPECT_EQ(plastic.hessian(point), NeoHookeanEnergyFp{}.hessian(inputs));
    EXPECT_EQ(parameters.hessian(point), NeoHookeanEnergyTheta{}.hessian(inputs));
    fields.F = ElementField<Eigen::Matrix3d>(1.2 * F);
    fields.Fp = ElementField<Eigen::Matrix3d>(0.9 * Fp);
    fields.theta = ConstantField<Eigen::Vector2d>(Eigen::Vector2d(4, 5));
    EXPECT_DOUBLE_EQ(deformation.value(point), expected);
    EXPECT_DOUBLE_EQ(plastic.value(point), expected);
    EXPECT_DOUBLE_EQ(parameters.value(point), expected);
    EXPECT_NE((NeoHookeanLocalEnergyF(fields).value(point)), expected);
    const auto owned = [&] {
        MaterialFields local{ElementField<Eigen::Matrix3d>(F), ElementField<Eigen::Matrix3d>(Fp),
                             ConstantField<Eigen::Vector2d>(Eigen::Vector2d(2, 3))};
        return NeoHookeanLocalEnergyF(std::move(local));
    }();
    EXPECT_DOUBLE_EQ(owned.value(point), expected);
}
} // namespace
