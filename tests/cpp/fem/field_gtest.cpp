#include "litepgo/fem/constant_field.hpp"
#include "litepgo/fem/deformation_gradient_field.hpp"
#include "litepgo/fem/element_field.hpp"
#include "litepgo/fem/field.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/interpolated_value_field.hpp"
#include "litepgo/fem/local_energy.hpp"
#include "litepgo/fem/neo_hookean_local_energy.hpp"
#include "litepgo/fem/volume_field_inputs.hpp"

#include <gtest/gtest.h>

namespace {
struct ValueOnlyParameters {
    using FieldValue = Eigen::Vector2d;
    template <class Point> FieldValue value(const Point&) const {
        return {2, 3};
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
    const litepgo::VolumeFieldInputs fields{litepgo::ElementField<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()),
                                            litepgo::ElementField<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()),
                                            ValueOnlyParameters{}};
    const litepgo::NeoHookeanLocalEnergyF deformation(fields);
    const litepgo::NeoHookeanLocalEnergyFp plastic(fields);
    static_assert(litepgo::LocalFirstOrderEnergy<decltype(deformation), int>);
    static_assert(litepgo::LocalSecondOrderEnergy<decltype(deformation), int>);
    static_assert(litepgo::LocalFirstOrderEnergy<decltype(plastic), int>);
    static_assert(litepgo::LocalSecondOrderEnergy<decltype(plastic), int>);
    EXPECT_DOUBLE_EQ(deformation.value(0), 0);
    EXPECT_TRUE(deformation.gradient(0).isZero());
    EXPECT_TRUE(deformation.hessian(0).allFinite());
    EXPECT_TRUE(plastic.gradient(0).isZero());
    EXPECT_TRUE(plastic.hessian(0).allFinite());
}
} // namespace
