#include "litepgo/fem/constant_field.hpp"
#include "litepgo/fem/deformation_gradient_field.hpp"
#include "litepgo/fem/element_field.hpp"
#include "litepgo/fem/field.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/interpolated_value_field.hpp"
#include "litepgo/fem/local_energy.hpp"
#include "litepgo/fem/material_fields.hpp"
#include "litepgo/fem/neo_hookean_local_energy.hpp"

#include <gtest/gtest.h>

namespace {
struct ValueOnlyParameters {
    using FieldValue = Eigen::Vector2d;
    template <class Point> FieldValue value(const Point&) const {
        return {2, 3};
    }
};
struct ValueOnlyDeformation {
    using FieldValue = Eigen::Matrix3d;
    static constexpr int num_dofs = 9;
    FieldValue value(const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData&) const {
        return Eigen::Matrix3d::Identity();
    }
};
struct WrongValue {
    using FieldValue = Eigen::Vector2d;
    double value(int) const;
};
struct WrongJacobian {
    using FieldValue = Eigen::Vector2d;
    static constexpr int num_dofs = 2;
    FieldValue value(int) const;
    Eigen::Matrix3d jacobian(int) const;
};
struct ZeroDofs {
    using FieldValue = Eigen::Vector2d;
    static constexpr int num_dofs = 0;
    FieldValue value(int) const;
    Eigen::Matrix<double, 2, 0> jacobian(int) const;
};
struct DynamicValue {
    using FieldValue = Eigen::VectorXd;
    FieldValue value(int) const;
};
static_assert(litepgo::Field<ValueOnlyParameters, int>);
static_assert(!litepgo::AffineField<ValueOnlyParameters, int>);
static_assert(!litepgo::Field<int, int>);
static_assert(!litepgo::Field<WrongValue, int>);
static_assert(!litepgo::Field<DynamicValue, int>);
static_assert(litepgo::Field<WrongJacobian, int>);
static_assert(!litepgo::AffineField<WrongJacobian, int>);
static_assert(!litepgo::AffineField<ZeroDofs, int>);
static_assert(!litepgo::Field<litepgo::ElementDofData<Eigen::Matrix3d>, int>);
static_assert(!litepgo::Field<litepgo::ConstantDofData<Eigen::Vector2d>, int>);
static_assert(litepgo::AffineField<litepgo::ElementField<Eigen::Matrix3d>, int>);
static_assert(litepgo::AffineField<litepgo::ConstantField<Eigen::Vector2d>, int>);
static_assert(litepgo::AffineField<litepgo::DeformationGradientField<litepgo::HexQ1Basis>,
                                   litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
static_assert(litepgo::AffineField<litepgo::InterpolatedValueField<litepgo::HexQ1Basis, 2>,
                                   litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
static_assert(!litepgo::Field<litepgo::DeformationGradientField<litepgo::HexQ1Basis>, int>);

TEST(Field, InactiveFieldNeedsOnlyValue) {
    Eigen::Matrix<double, 8, 3> rest;
    for (int i = 0; i < 8; ++i)
        rest.row(i) = litepgo::HexQ1Basis::reference_node(i).transpose();
    const litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis> geometry(rest);
    const auto point = geometry.evaluate({0.2, 0.3, 0.4});

    const litepgo::MaterialFields fields{litepgo::ElementField<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()),
                                         litepgo::ElementField<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()),
                                         ValueOnlyParameters{}};
    const litepgo::NeoHookeanLocalEnergyF deformation(fields);
    const litepgo::NeoHookeanLocalEnergyFp plastic(fields);
    static_assert(litepgo::LocalFirstOrderEnergy<decltype(deformation),
                                                 litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
    static_assert(litepgo::LocalSecondOrderEnergy<decltype(deformation),
                                                  litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
    static_assert(litepgo::LocalFirstOrderEnergy<decltype(plastic),
                                                 litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
    static_assert(litepgo::LocalSecondOrderEnergy<decltype(plastic),
                                                  litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
    EXPECT_DOUBLE_EQ(deformation.value(point), 0);
    EXPECT_TRUE(deformation.gradient(point).isZero());
    EXPECT_TRUE(deformation.hessian(point).allFinite());
    EXPECT_TRUE(plastic.gradient(point).isZero());
    EXPECT_TRUE(plastic.hessian(point).allFinite());
    const litepgo::MaterialFields value_fields{ValueOnlyDeformation{}, fields.Fp, fields.theta};
    const litepgo::NeoHookeanLocalEnergyF value_only(value_fields);
    static_assert(
        litepgo::LocalValue<decltype(value_only), litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
    static_assert(!litepgo::LocalFirstOrderEnergy<decltype(value_only),
                                                  litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
    static_assert(!litepgo::LocalSecondOrderEnergy<decltype(value_only),
                                                   litepgo::VolumeReferenceGeometry<litepgo::HexQ1Basis>::PointData>);
    EXPECT_DOUBLE_EQ(value_only.value(point), 0);
}
} // namespace
