#pragma once

#include "litepgo/fem/field/field.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace litepgo {

// Spatially constant; coefficients and values use the same component vector.
template <int ParamDim> class IdentityFieldT final : public FieldT<ParamDim> {
    using Base = FieldT<ParamDim>;

  public:
    using typename Base::Coordinate;
    using typename Base::SpatialDerivative;
    using typename Base::SpatialHessian;

    explicit IdentityFieldT(Eigen::VectorXd coeffs) : coeffs_(std::move(coeffs)) {
        if (coeffs_.size() == 0 || coeffs_.size() > std::numeric_limits<int>::max() || !coeffs_.allFinite())
            throw std::invalid_argument("IdentityFieldT requires nonempty finite values with int-sized DOFs");
    }
    int num_dofs() const override {
        return static_cast<int>(coeffs_.size());
    }
    int num_components() const override {
        return static_cast<int>(coeffs_.size());
    }
    Eigen::VectorXd value(const Coordinate& xi) const override {
        check_xi(xi);
        return coeffs_;
    }
    SpatialDerivative dvalue_dxi(const Coordinate& xi) const override {
        check_xi(xi);
        return SpatialDerivative::Zero(coeffs_.size(), ParamDim);
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Coordinate& xi) const override {
        check_xi(xi);
        return Eigen::MatrixXd::Identity(coeffs_.size(), coeffs_.size());
    }
    Eigen::MatrixXd dgrad_xi_dcoeffs(const Coordinate& xi) const override {
        check_xi(xi);
        return Eigen::MatrixXd::Zero(ParamDim * coeffs_.size(), coeffs_.size());
    }
    SpatialHessian d2value_dxi2(const Coordinate& xi) const override {
        check_xi(xi);
        return SpatialHessian::Zero(coeffs_.size(), Base::num_second_derivatives);
    }
    Eigen::MatrixXd dhess_xi_dcoeffs(const Coordinate& xi) const override {
        check_xi(xi);
        return Eigen::MatrixXd::Zero(Base::num_second_derivatives * coeffs_.size(), coeffs_.size());
    }

  private:
    static void check_xi(const Coordinate& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("IdentityFieldT requires finite parameters");
    }
    Eigen::VectorXd coeffs_;
};

using IdentityField2D = IdentityFieldT<2>;
using IdentityField3D = IdentityFieldT<3>;

} // namespace litepgo
