#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include <gtest/gtest.h>
#include <type_traits>

using namespace litepgo;
TEST(FieldSample, SnapshotsAreReadOnlyOwningAndNotPubliclyConstructible) {
    using Field = InterpolatedField<HexQ1Basis>;
    static_assert(!std::is_default_constructible_v<Field::Sample>);
    static_assert(std::is_const_v<decltype(std::declval<Field::Sample>().value)>);
    static_assert(std::is_const_v<decltype(std::declval<Field::Sample>().xi)>);
    const auto field_sample = [] { return Field(Eigen::MatrixXd::Constant(8, 2, 3)).evaluate({0.2, 0.3, 0.4}); }();
    EXPECT_TRUE(field_sample.value.isApprox(Eigen::Vector2d(3, 3), 1e-14));
    EXPECT_EQ(field_sample.xi, Eigen::Vector3d(0.2, 0.3, 0.4));
    EXPECT_EQ(field_sample.dvalue_dcoeffs().cols(), 16);
    EXPECT_EQ(field_sample.dgrad_xi_dcoeffs().rows(), 6);
}
TEST(FieldSample, IdentityFieldHasZeroSpatialAndMixedDerivatives) {
    const auto field_sample = IdentityField<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()).evaluate({0.2, 0.3, 0.4});
    EXPECT_EQ(field_sample.dvalue_dxi.rows(), 9);
    EXPECT_TRUE(field_sample.dvalue_dxi.isZero());
    EXPECT_TRUE(field_sample.dvalue_dcoeffs().isIdentity());
    EXPECT_EQ(field_sample.dgrad_xi_dcoeffs().rows(), 27);
    EXPECT_TRUE(field_sample.dgrad_xi_dcoeffs().isZero());
}
