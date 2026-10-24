#include "litepgo/fem/constant_field.hpp"
#include "litepgo/fem/element_field.hpp"
#include "litepgo/fem/neo_hookean_local_energy.hpp"
#include "litepgo/fem/volume_field_inputs.hpp"

#include <gtest/gtest.h>

namespace {
using namespace litepgo;
template <class Fields>
concept CanBindAll = requires {
    typename NeoHookeanLocalEnergyF<Fields>;
    typename NeoHookeanLocalEnergyFp<Fields>;
    typename NeoHookeanLocalEnergyTheta<Fields>;
};
static_assert(CanBindAll<VolumeFieldInputs<ElementField<Mat3>, ElementField<Mat3>, ConstantField<Eigen::Vector2d>>>);
static_assert(!CanBindAll<VolumeFieldInputs<ElementField<Mat3>, ElementField<Mat3>, ConstantField<Eigen::Vector3d>>>);

TEST(NeoHookeanLocalEnergy, DirectFieldsBindCorrectDerivativesAndOwnSnapshots) {
    Mat3 F, Fp;
    F << 1.2, 0.1, -0.03, 0.04, 0.9, 0.02, -0.01, 0.04, 1.1;
    Fp << 0.95, -0.02, 0.05, 0.01, 1.1, -0.04, 0.03, 0.01, 1.05;
    VolumeFieldInputs fields{ElementField<Mat3>(F), ElementField<Mat3>(Fp),
                             ConstantField<Eigen::Vector2d>(Eigen::Vector2d(2, 3))};
    const NeoHookeanLocalEnergyF deformation(fields);
    const NeoHookeanLocalEnergyFp plastic(fields);
    const NeoHookeanLocalEnergyTheta parameters(fields);
    static_assert(decltype(deformation)::num_dofs == 9);
    static_assert(decltype(plastic)::num_dofs == 9);
    static_assert(decltype(parameters)::num_dofs == 2);
    static_assert(!std::is_constructible_v<decltype(deformation), decltype(fields), NeoHookeanEnergyFp>);
    const auto inputs = fields.evaluate(0.3);
    const double expected = NeoHookean{}.value(inputs);
    EXPECT_DOUBLE_EQ(deformation.value(0.3), expected);
    EXPECT_DOUBLE_EQ(plastic.value(0.7), expected);
    EXPECT_DOUBLE_EQ(parameters.value(0.9), expected);
    const auto stress = NeoHookeanEnergyF{}.gradient(inputs);
    const auto plastic_stress = NeoHookeanEnergyFp{}.gradient(inputs);
    EXPECT_TRUE(
        deformation.gradient(0.3).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(stress.data()), 1e-13));
    EXPECT_TRUE(
        plastic.gradient(0.3).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(plastic_stress.data()), 1e-13));
    EXPECT_EQ(parameters.gradient(0.3), NeoHookeanEnergyTheta{}.gradient(inputs));
    EXPECT_EQ(deformation.hessian(0.3), NeoHookeanEnergyF{}.hessian(inputs));
    EXPECT_EQ(plastic.hessian(0.3), NeoHookeanEnergyFp{}.hessian(inputs));
    EXPECT_EQ(parameters.hessian(0.3), NeoHookeanEnergyTheta{}.hessian(inputs));
    fields.F = ElementField<Mat3>(1.2 * F);
    fields.Fp = ElementField<Mat3>(0.9 * Fp);
    fields.theta = ConstantField<Eigen::Vector2d>(Eigen::Vector2d(4, 5));
    EXPECT_DOUBLE_EQ(deformation.value(0.3), expected);
    EXPECT_DOUBLE_EQ(plastic.value(0.3), expected);
    EXPECT_DOUBLE_EQ(parameters.value(0.3), expected);
    EXPECT_NE(NeoHookeanLocalEnergyF(fields).value(0.3), expected);
    const auto owned = [&] {
        VolumeFieldInputs local{ElementField<Mat3>(F), ElementField<Mat3>(Fp),
                                ConstantField<Eigen::Vector2d>(Eigen::Vector2d(2, 3))};
        return NeoHookeanLocalEnergyF(std::move(local));
    }();
    EXPECT_DOUBLE_EQ(owned.value(0.3), expected);
}
} // namespace
