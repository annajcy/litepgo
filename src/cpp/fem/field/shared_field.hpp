#pragma once

#include "litepgo/fem/field/field.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace litepgo::fem {

// Type-erased field sharing ownership of any concrete field. Every evaluation is
// forwarded, so templates taking a field type (such as VolumetricFemEnergy) can use
// fields chosen at run time. Copies share the field, which must support concurrent
// const calls.
template <int ParamDim> class SharedFieldT final : public FieldT<ParamDim> {
    using Base = FieldT<ParamDim>;

  public:
    using typename Base::Coordinate;
    using typename Base::SpatialDerivative;
    using typename Base::SpatialHessian;

    explicit SharedFieldT(std::shared_ptr<const Base> field) : field_(std::move(field)) {
        if (!field_)
            throw std::invalid_argument("SharedFieldT requires a field");
    }

    int num_components() const override {
        return field_->num_components();
    }
    int num_dofs() const override {
        return field_->num_dofs();
    }
    Eigen::VectorXd eval_value(const Coordinate& xi) const override {
        return field_->eval_value(xi);
    }
    Eigen::MatrixXd eval_dvalue_dcoeffs(const Coordinate& xi) const override {
        return field_->eval_dvalue_dcoeffs(xi);
    }
    SpatialDerivative eval_dvalue_dxi(const Coordinate& xi) const override {
        return field_->eval_dvalue_dxi(xi);
    }
    Eigen::MatrixXd eval_dgrad_xi_dcoeffs(const Coordinate& xi) const override {
        return field_->eval_dgrad_xi_dcoeffs(xi);
    }
    SpatialHessian eval_d2value_dxi2(const Coordinate& xi) const override {
        return field_->eval_d2value_dxi2(xi);
    }
    Eigen::MatrixXd eval_dhess_xi_dcoeffs(const Coordinate& xi) const override {
        return field_->eval_dhess_xi_dcoeffs(xi);
    }

  private:
    std::shared_ptr<const Base> field_;
};

using SharedField2D = SharedFieldT<2>;
using SharedField3D = SharedFieldT<3>;

} // namespace litepgo::fem
