#include "litepgo/energy/response/composed_response.hpp"
#include <stdexcept>
namespace litepgo::energy {
ComposedResponse::ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner)
    : children_{std::move(outer), std::move(inner)} {
    if (!children_[0] || !children_[1])
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    if (children_[0]->num_inputs() != children_[1]->num_outputs())
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    num_inputs_ = children_[1]->num_inputs();
    num_outputs_ = children_[0]->num_outputs();
}
void ComposedResponse::check_children() const {
    if (!children_[0] || !children_[1])
        throw std::logic_error("ComposedResponse: cleared children");
}
bool ComposedResponse::has_hessian() const {
    check_children();
    return children_[0]->has_hessian() && children_[1]->has_hessian();
}
Vec ComposedResponse::eval_value_impl(const ConstVecRef& x) {
    check_children();
    return children_[0]->eval_value(children_[1]->eval_value(x));
}
std::pair<Vec, SpMat> ComposedResponse::eval_value_and_jacobian_impl(const ConstVecRef& x) {
    check_children();
    auto [y, inner_J] = children_[1]->eval_value_and_jacobian(x);
    auto [value, outer_J] = children_[0]->eval_value_and_jacobian(y);
    return {std::move(value), SpMat(outer_J * inner_J)};
}
SpMat ComposedResponse::eval_weighted_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights) {
    check_children();
    auto [y, J] = children_[1]->eval_value_and_jacobian(x);
    SpMat outer_H = children_[0]->eval_hessian(y, weights);
    Vec inner_weights = children_[0]->eval_vjp(y, weights);
    SpMat inner_H = children_[1]->eval_hessian(x, inner_weights);
    return SpMat(J.transpose() * outer_H * J + inner_H);
}
Vec ComposedResponse::eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& v) {
    check_children();
    const Vec y = children_[1]->eval_value(x);
    const Vec tangent = children_[1]->eval_jvp(x, v);
    return children_[0]->eval_jvp(y, tangent);
}
Vec ComposedResponse::eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& w) {
    check_children();
    const Vec y = children_[1]->eval_value(x);
    const Vec adjoint = children_[0]->eval_vjp(y, w);
    return children_[1]->eval_vjp(x, adjoint);
}
} // namespace litepgo::energy
