#pragma once

#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/fem/basis/basis.hpp"
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace litepgo::fem {

// R(xi) = C^T N(xi). Coefficients are fixed snapshots; all derivatives are with respect to xi.
template <class B> class InterpolatedResponse final : public energy::FixedPatternResponse {
    static constexpr int dimension = std::tuple_size_v<decltype(B::coefficient_info(0).derivative_order)>;
    static_assert(Basis<B, dimension>);

  public:
    // Rows follow basis coefficient order; columns are output components.
    explicit InterpolatedResponse(Eigen::MatrixXd coefficients)
        : FixedPatternResponse(checked_pattern(coefficients), Eigen::MatrixXd::Ones(dimension, dimension).sparseView()),
          coefficients_(std::move(coefficients)) {}

  private:
    static SpMat checked_pattern(const Eigen::MatrixXd& C) {
        if (C.rows() != B::num_basis_functions || C.cols() <= 0 || C.cols() > std::numeric_limits<int>::max() ||
            !C.allFinite())
            throw std::invalid_argument("InterpolatedResponse: incompatible or non-finite coefficients");
        return Eigen::MatrixXd::Ones(C.cols(), dimension).sparseView();
    }
    static auto coordinate(const ConstVecRef& xi) {
        if constexpr (dimension == 1)
            return xi[0];
        else
            return Eigen::Matrix<double, dimension, 1>(xi);
    }
    Vec eval_impl(const ConstVecRef& xi, SpMat* J) override {
        const auto point = coordinate(xi);
        if (J) {
            const Eigen::MatrixXd derivative = coefficients_.transpose() * B::eval_dN_dxi(point);
            for (int c = 0; c < dimension; ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = derivative(it.row(), c);
        }
        return coefficients_.transpose() * B::eval_N(point);
    }
    Vec eval_jvp_impl(const ConstVecRef& xi, const ConstVecRef& v) override {
        return coefficients_.transpose() * (B::eval_dN_dxi(coordinate(xi)) * v);
    }
    Vec eval_vjp_impl(const ConstVecRef& xi, const ConstVecRef& w) override {
        return B::eval_dN_dxi(coordinate(xi)).transpose() * (coefficients_ * w);
    }
    void eval_hessian_impl(const ConstVecRef& xi, const ConstVecRef& w, SpMat& H) override {
        const Vec packed = B::eval_d2N_dxi2(coordinate(xi)).transpose() * (coefficients_ * w);
        // Basis stores each upper-triangle entry once, without scaling mixed terms.
        int k = 0;
        for (int i = 0; i < dimension; ++i)
            for (int j = i; j < dimension; ++j, ++k) {
                H.coeffRef(i, j) = packed[k];
                H.coeffRef(j, i) = packed[k];
            }
    }
    Eigen::MatrixXd coefficients_;
};
} // namespace litepgo::fem
