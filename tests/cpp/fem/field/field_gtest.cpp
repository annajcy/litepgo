#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/field/identity_field.hpp"

#include <gtest/gtest.h>
#include <type_traits>

namespace {
using namespace litepgo;

template <class T>
constexpr bool has_legacy_geometry_api = requires(const T& field, const Eigen::Vector3d& xi) { field.dX_dxi(xi); } ||
                                         requires(const T& field, const Eigen::Vector3d& xi) { field.dxi_dX(xi); };
static_assert(!has_legacy_geometry_api<Field3D>);

class CoefficientOnlyField final : public Field3D {
  public:
    int num_components() const override {
        return 2;
    }
    int num_dofs() const override {
        return 2;
    }
    Eigen::VectorXd value(const Eigen::Vector3d&) const override {
        return Eigen::Vector2d(2, 3);
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d&) const override {
        return Eigen::Matrix2d::Identity();
    }
};

TEST(Field3D, UnsupportedSpatialDerivativesThrowThroughBaseReference) {
    static_assert(std::is_abstract_v<Field3D>);
    static_assert(std::has_virtual_destructor_v<Field3D>);
    const CoefficientOnlyField concrete;
    const Field3D& field = concrete;
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_EQ(field.value(xi), Eigen::Vector2d(2, 3));
    EXPECT_TRUE(field.dvalue_dcoeffs(xi).isIdentity());
    EXPECT_THROW(field.dvalue_dxi(xi), std::logic_error);
    EXPECT_THROW(field.dgrad_xi_dcoeffs(xi), std::logic_error);
    EXPECT_THROW(field.d2value_dxi2(xi), std::logic_error);
    EXPECT_THROW(field.dhess_xi_dcoeffs(xi), std::logic_error);
}

TEST(Field3D, IdentityFieldDispatchesValuesAndDerivativesThroughBaseReference) {
    const IdentityField3D concrete(Eigen::Matrix3d::Identity().reshaped());
    const Field3D& field = concrete;
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_EQ(field.num_components(), 9);
    EXPECT_EQ(field.num_dofs(), 9);
    EXPECT_EQ(field.value(xi), Eigen::Matrix3d::Identity().reshaped());
    EXPECT_TRUE(field.dvalue_dcoeffs(xi).isIdentity());
    EXPECT_EQ(field.dvalue_dxi(xi).rows(), 9);
    EXPECT_TRUE(field.dvalue_dxi(xi).isZero());
    EXPECT_EQ(field.dgrad_xi_dcoeffs(xi).rows(), 27);
    EXPECT_EQ(field.dgrad_xi_dcoeffs(xi).cols(), 9);
    EXPECT_TRUE(field.dgrad_xi_dcoeffs(xi).isZero());
}
} // namespace
