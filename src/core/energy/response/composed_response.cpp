#include "litepgo/energy/response/composed_response.hpp"
#include <stdexcept>
namespace litepgo::energy {
ComposedResponse::ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner)
    : outer_(std::move(outer)), inner_(std::move(inner)) {
    if (!outer_ || !inner_)
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    if (outer_->num_inputs() != inner_->num_outputs())
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    num_inputs_ = inner_->num_inputs();
    num_outputs_ = outer_->num_outputs();
}
bool ComposedResponse::has_hessian() const {
    return outer_->has_hessian() && inner_->has_hessian();
}
Vec ComposedResponse::eval_value_impl(const ConstVecRef& x) {
    return outer_->eval_value(inner_->eval_value(x));
}
std::pair<Vec, SpMat> ComposedResponse::eval_value_and_jacobian_impl(const ConstVecRef& x) {
    auto [y, inner_J] = inner_->eval_value_and_jacobian(x);
    auto [value, outer_J] = outer_->eval_value_and_jacobian(y);
    return {std::move(value), SpMat(outer_J * inner_J)};
}
SpMat ComposedResponse::eval_weighted_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights) {
    auto [y, J] = inner_->eval_value_and_jacobian(x);
    SpMat outer_H = outer_->eval_hessian(y, weights);
    Vec inner_weights = outer_->eval_vjp(y, weights);
    SpMat inner_H = inner_->eval_hessian(x, inner_weights);
    return SpMat(J.transpose() * outer_H * J + inner_H);
}
Vec ComposedResponse::eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& v) {
    const Vec y = inner_->eval_value(x);
    const Vec tangent = inner_->eval_jvp(x, v);
    return outer_->eval_jvp(y, tangent);
}
Vec ComposedResponse::eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& w) {
    const Vec y = inner_->eval_value(x);
    const Vec adjoint = outer_->eval_vjp(y, w);
    return inner_->eval_vjp(x, adjoint);
}
} // namespace litepgo::energy
