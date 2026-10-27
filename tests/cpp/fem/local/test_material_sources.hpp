#pragma once

#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"

#include <utility>

namespace litepgo::test {
// Fixtures returned by value own their inputs; each operation creates a short-lived borrowing field.
template <class UField, class XField> class OwnedDeformationField final : public Field {
  public:
    OwnedDeformationField(UField u_field, XField X_field)
        : u_field_(std::move(u_field)), X_field_(std::move(X_field)) {}
    int num_components() const override {
        return 9;
    }
    int num_dofs() const override {
        return u_field_.num_dofs();
    }
    const Field& coordinate_field() const {
        return X_field_;
    }
    Eigen::VectorXd value(const Eigen::Vector3d& xi) const override {
        return DeformationGradientField(u_field_, X_field_).value(xi);
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d& xi) const override {
        return DeformationGradientField(u_field_, X_field_).dvalue_dcoeffs(xi);
    }

  private:
    UField u_field_;
    XField X_field_;
};
template <class FSource, class FpSource, class ThetaSource> struct MaterialSources {
    FSource F;
    FpSource Fp;
    ThetaSource theta;
    MaterialInputs evaluate(const Eigen::Vector3d& xi) const {
        return {F.value(xi).reshaped(3, 3), Fp.value(xi).reshaped(3, 3), theta.value(xi)};
    }
};
template <class Energy, class Sources> auto bind_energy(const Sources& fields) {
    return Energy(fields.F, fields.Fp, fields.theta);
}
} // namespace litepgo::test
