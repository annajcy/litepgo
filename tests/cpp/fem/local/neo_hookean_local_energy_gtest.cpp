#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/local/neo_hookean_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"

#include "litepgo/fem/geometry/volume_reference_geometry.hpp"
#include <gtest/gtest.h>

namespace {
using namespace litepgo;
TEST(NeoHookeanLocalEnergy, DirectFieldsBindCorrectDerivativesAndOwnSnapshots) {
    Eigen::Matrix<double, 8, 3> rest;
    for (int i = 0; i < 8; ++i)
        rest.row(i) = HexQ1Basis::reference_node(i).transpose();
    const VolumeReferenceGeometry<HexQ1Basis> geometry(rest);
    const auto point = geometry.evaluate({0.2, 0.3, 0.4});

    Eigen::Matrix3d F, Fp;
    F << 1.2, 0.1, -0.03, 0.04, 0.9, 0.02, -0.01, 0.04, 1.1;
    Fp << 0.95, -0.02, 0.05, 0.01, 1.1, -0.04, 0.03, 0.01, 1.05;
    MaterialFields fields{std::make_shared<IdentityField<Eigen::Matrix3d>>(F),
                          std::make_shared<IdentityField<Eigen::Matrix3d>>(Fp),
                          std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3))};
    const NeoHookeanLocalEnergyF deformation(fields);
    const NeoHookeanLocalEnergyFp plastic(fields);
    const NeoHookeanLocalEnergyTheta parameters(fields);
    for (const LocalEnergy* energy :
         {static_cast<const LocalEnergy*>(&deformation), static_cast<const LocalEnergy*>(&plastic),
          static_cast<const LocalEnergy*>(&parameters)}) {
        EXPECT_TRUE(energy->has_gradient());
        EXPECT_TRUE(energy->has_hessian());
    }

    const MaterialFields nodal_fields{std::make_shared<DeformationGradientField>(Eigen::Matrix<double, 8, 3>::Zero()),
                                      fields.Fp, fields.theta};
    const NeoHookeanLocalEnergyF nodal_deformation(nodal_fields);
    const NeoHookeanLocalEnergyFp nodal_plastic(nodal_fields);
    const NeoHookeanLocalEnergyTheta nodal_parameters(nodal_fields);
    EXPECT_EQ(nodal_deformation.num_dofs(), 24);
    EXPECT_EQ(nodal_plastic.num_dofs(), 9);
    EXPECT_EQ(nodal_parameters.num_dofs(), 2);
    EXPECT_EQ(nodal_deformation.gradient(point).size(), 24);
    EXPECT_EQ(nodal_plastic.hessian(point).rows(), 9);
    EXPECT_EQ(nodal_parameters.hessian(point).rows(), 2);
    EXPECT_THROW(nodal_deformation.value(VolumePointData{}), std::invalid_argument);
    EXPECT_THROW(nodal_plastic.gradient(VolumePointData{}), std::invalid_argument);
    EXPECT_THROW(nodal_parameters.hessian(VolumePointData{}), std::invalid_argument);
    const auto inputs = fields.evaluate(point);
    const double expected = NeoHookean{}.value(inputs);
    EXPECT_DOUBLE_EQ(deformation.value(point), expected);
    EXPECT_DOUBLE_EQ(plastic.value(point), expected);
    EXPECT_DOUBLE_EQ(parameters.value(point), expected);
    EXPECT_DOUBLE_EQ(deformation.value(VolumePointData{}), expected);
    EXPECT_EQ(plastic.gradient(VolumePointData{}), plastic.gradient(point));
    EXPECT_EQ(parameters.hessian(VolumePointData{}), parameters.hessian(point));
    const auto stress = NeoHookean{}.gradient_F(inputs);
    const auto plastic_stress = NeoHookean{}.gradient_Fp(inputs);
    EXPECT_TRUE(
        deformation.gradient(point).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(stress.data()), 1e-13));
    EXPECT_TRUE(
        plastic.gradient(point).isApprox(Eigen::Map<const Eigen::Matrix<double, 9, 1>>(plastic_stress.data()), 1e-13));
    EXPECT_EQ(parameters.gradient(point), NeoHookean{}.gradient_theta(inputs));
    EXPECT_EQ(deformation.hessian(point), NeoHookean{}.hessian_F(inputs));
    EXPECT_EQ(plastic.hessian(point), NeoHookean{}.hessian_Fp(inputs));
    EXPECT_EQ(parameters.hessian(point), NeoHookean{}.hessian_theta(inputs));
    fields.F = std::make_shared<IdentityField<Eigen::Matrix3d>>(1.2 * F);
    fields.Fp = std::make_shared<IdentityField<Eigen::Matrix3d>>(0.9 * Fp);
    fields.theta = std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(4, 5));
    EXPECT_DOUBLE_EQ(deformation.value(point), expected);
    EXPECT_DOUBLE_EQ(plastic.value(point), expected);
    EXPECT_DOUBLE_EQ(parameters.value(point), expected);
    EXPECT_NE((NeoHookeanLocalEnergyF(fields).value(point)), expected);
    const auto owned = [&] {
        MaterialFields local{std::make_shared<IdentityField<Eigen::Matrix3d>>(F),
                             std::make_shared<IdentityField<Eigen::Matrix3d>>(Fp),
                             std::make_shared<IdentityField<Eigen::VectorXd>>(Eigen::Vector2d(2, 3))};
        return NeoHookeanLocalEnergyF(std::move(local));
    }();
    EXPECT_DOUBLE_EQ(owned.value(point), expected);
}
} // namespace
