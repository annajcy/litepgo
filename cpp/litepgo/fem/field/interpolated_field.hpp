#pragma once

#include "litepgo/fem/basis/basis.hpp"
#include "litepgo/fem/field/field.hpp"

#include <limits>
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

// Scalar coefficients flatten by row: m*a+c. Reference derivatives have m rows and ParamDim columns.
template <class B, int ParamDim>
    requires Basis<B, ParamDim>
class InterpolatedFieldT final : public FieldT<ParamDim> {
    using Base = FieldT<ParamDim>;

  public:
    using typename Base::Coordinate;
    using typename Base::SpatialDerivative;
    using typename Base::SpatialHessian;
    static constexpr int num_basis_functions = B::num_basis_functions;
    explicit InterpolatedFieldT(Eigen::MatrixXd coeffs) : coeffs_(std::move(coeffs)) {
        if (coeffs_.rows() != num_basis_functions || coeffs_.cols() == 0 ||
            coeffs_.size() > std::numeric_limits<int>::max() || !coeffs_.allFinite())
            throw std::invalid_argument(
                "InterpolatedFieldT requires compatible finite coefficients with int-sized DOFs");
    }
    int num_dofs() const override {
        return static_cast<int>(coeffs_.size());
    }
    int num_components() const override {
        return static_cast<int>(coeffs_.cols());
    }

    Eigen::VectorXd value(const Coordinate& xi) const override {
        const auto N = checked_basis_sample(xi).N;
        Eigen::VectorXd result = coeffs_.transpose() * N;
        if (!N.allFinite() || !result.allFinite())
            throw std::invalid_argument("InterpolatedFieldT::value: non-finite data");
        return result;
    }
    SpatialDerivative dvalue_dxi(const Coordinate& xi) const override {
        const auto dN_dxi = checked_basis_sample(xi).dN_dxi;
        SpatialDerivative result = coeffs_.transpose() * dN_dxi;
        if (!dN_dxi.allFinite() || !result.allFinite())
            throw std::invalid_argument("InterpolatedFieldT::dvalue_dxi: non-finite data");
        return result;
    }
    Eigen::MatrixXd dvalue_dcoeffs(const Coordinate& xi) const override {
        const auto N = checked_basis_sample(xi).N;
        if (!N.allFinite())
            throw std::invalid_argument("InterpolatedFieldT::dvalue_dcoeffs: non-finite basis values");
        return Eigen::kroneckerProduct(N.transpose(), Eigen::MatrixXd::Identity(num_components(), num_components()));
    }
    Eigen::MatrixXd dgrad_xi_dcoeffs(const Coordinate& xi) const override {
        const auto dN_dxi = checked_basis_sample(xi).dN_dxi;
        if (!dN_dxi.allFinite())
            throw std::invalid_argument("InterpolatedFieldT::dgrad_xi_dcoeffs: non-finite basis gradients");
        return Eigen::kroneckerProduct(dN_dxi.transpose(),
                                       Eigen::MatrixXd::Identity(num_components(), num_components()));
    }
    SpatialHessian d2value_dxi2(const Coordinate& xi) const override {
        if constexpr (SecondOrderBasis<B, ParamDim>) {
            const auto d2N_dxi2 = checked_basis_sample(xi).d2N_dxi2;
            SpatialHessian result = coeffs_.transpose() * d2N_dxi2;
            if (!d2N_dxi2.allFinite() || !result.allFinite())
                throw std::invalid_argument("InterpolatedFieldT::d2value_dxi2: non-finite data");
            return result;
        } else {
            return Base::d2value_dxi2(xi);
        }
    }
    Eigen::MatrixXd dhess_xi_dcoeffs(const Coordinate& xi) const override {
        if constexpr (SecondOrderBasis<B, ParamDim>) {
            const auto d2N_dxi2 = checked_basis_sample(xi).d2N_dxi2;
            if (!d2N_dxi2.allFinite())
                throw std::invalid_argument("InterpolatedFieldT::dhess_xi_dcoeffs: non-finite basis Hessians");
            return Eigen::kroneckerProduct(d2N_dxi2.transpose(),
                                           Eigen::MatrixXd::Identity(num_components(), num_components()));
        } else {
            return Base::dhess_xi_dcoeffs(xi);
        }
    }

  private:
    static typename B::Sample checked_basis_sample(const Coordinate& xi) {
        if (!xi.allFinite())
            throw std::invalid_argument("InterpolatedFieldT requires finite parameters");
        return B::evaluate(xi);
    }
    Eigen::MatrixXd coeffs_;
};

template <Basis2D B> using InterpolatedField2D = InterpolatedFieldT<B, 2>;
template <Basis3D B> using InterpolatedField3D = InterpolatedFieldT<B, 3>;

} // namespace litepgo
