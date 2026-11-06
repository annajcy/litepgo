#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <stdexcept>

namespace litepgo::energy {
namespace {
SpMat product_pattern(SpMat outer, SpMat inner) {
    outer.coeffs().setOnes();
    inner.coeffs().setOnes();
    return linalg::pattern_of(SpMat(outer * inner));
}
SpMat composed_hessian_pattern(SpMat inner_J, SpMat inner_H, SpMat outer_H) {
    inner_J.coeffs().setOnes();
    inner_H.coeffs().setOnes();
    outer_H.coeffs().setOnes();
    return linalg::pattern_of(SpMat(inner_J.transpose() * outer_H * inner_J + inner_H));
}
std::optional<SpMat> initial_hessian_pattern(const std::shared_ptr<Response>& outer,
                                             const std::shared_ptr<Response>& inner) {
    if (!outer || !inner || outer->num_inputs() != inner->num_outputs() || !outer->has_hessian() ||
        !inner->has_hessian())
        return std::nullopt;
    return composed_hessian_pattern(inner->jacobian_pattern(), inner->hessian_pattern(), outer->hessian_pattern());
}
SpMat initial_pattern(const std::shared_ptr<Response>& outer, const std::shared_ptr<Response>& inner) {
    if (!outer || !inner || outer->num_inputs() != inner->num_outputs())
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    return product_pattern(outer->jacobian_pattern(), inner->jacobian_pattern());
}
} // namespace
ComposedResponse::ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner)
    : Response(initial_pattern(outer, inner),
               outer && inner && outer->has_fixed_pattern() && inner->has_fixed_pattern(),
               initial_hessian_pattern(outer, inner)),
      children_{std::move(outer), std::move(inner)} {}
void ComposedResponse::check_children() const {
    if (!children_[0] || !children_[1])
        throw std::logic_error("ComposedResponse: cleared children");
}
std::optional<SpMat> ComposedResponse::refresh_pattern(const ConstVecRef& x) {
    check_children();
    children_[1]->update_pattern(x);
    const Vec y = children_[1]->eval_value(x);
    // Snapshot before updating outer: both roles may share a dynamic child.
    const SpMat inner_pattern = children_[1]->jacobian_pattern();
    children_[0]->update_pattern(y);
    return product_pattern(children_[0]->jacobian_pattern(), inner_pattern);
}
std::optional<SpMat> ComposedResponse::refresh_hessian_pattern(const ConstVecRef& x) {
    check_children();
    children_[1]->update_pattern(x);
    const Vec y = children_[1]->eval_value(x);
    const SpMat inner_J = children_[1]->jacobian_pattern();
    const SpMat inner_H = children_[1]->hessian_pattern();
    children_[0]->update_pattern(y);
    return composed_hessian_pattern(inner_J, inner_H, children_[0]->hessian_pattern());
}
void ComposedResponse::eval_hessian_values(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) const {
    check_children();
    auto& inner = *children_[1];
    auto& outer = *children_[0];
    inner.update_pattern(x);
    const Vec y = inner.eval_value(x);
    const SpMat inner_J = inner.eval_jacobian(x);
    const SpMat inner_H_pattern = inner.hessian_pattern();
    outer.update_pattern(y);
    if (!linalg::same_pattern(composed_hessian_pattern(inner_J, inner_H_pattern, outer.hessian_pattern()),
                              hessian_pattern()))
        throw std::logic_error("ComposedResponse: call update_pattern on the composition");
    const SpMat outer_H = outer.eval_hessian(y, weights);
    const Vec inner_weights = outer.eval_vjp(y, weights);
    // The same dynamic child may appear in both roles; restore its inner state.
    inner.update_pattern(x);
    const SpMat inner_H = inner.eval_hessian(x, inner_weights);
    const SpMat result = inner_J.transpose() * outer_H * inner_J + inner_H;
    for (int c = 0; c < H.outerSize(); ++c)
        for (SpMat::InnerIterator it(H, c); it; ++it)
            it.valueRef() = result.coeff(it.row(), c);
}
Vec ComposedResponse::eval_values(const ConstVecRef& x, SpMat* J) const {
    check_children();
    if (!J)
        return children_[0]->eval_value(children_[1]->eval_value(x));
    children_[1]->update_pattern(x);
    SpMat inner_J = children_[1]->jacobian_pattern();
    const Vec y = children_[1]->eval(x, &inner_J);
    children_[0]->update_pattern(y);
    SpMat outer_J = children_[0]->jacobian_pattern();
    if (!linalg::same_pattern(product_pattern(outer_J, inner_J), jacobian_pattern()))
        throw std::logic_error("ComposedResponse: call update_pattern on the composition");
    Vec value = children_[0]->eval(y, &outer_J);
    const SpMat product = outer_J * inner_J;
    for (int c = 0; c < J->outerSize(); ++c)
        for (SpMat::InnerIterator it(*J, c); it; ++it)
            it.valueRef() = product.coeff(it.row(), c);
    return value;
}
Vec ComposedResponse::eval_jvp_values(const ConstVecRef& x, const ConstVecRef& v) const {
    check_children();
    children_[1]->update_pattern(x);
    const Vec y = children_[1]->eval_value(x);
    const Vec tangent = children_[1]->eval_jvp(x, v);
    children_[0]->update_pattern(y);
    return children_[0]->eval_jvp(y, tangent);
}
Vec ComposedResponse::eval_vjp_values(const ConstVecRef& x, const ConstVecRef& w) const {
    check_children();
    const Vec y = children_[1]->eval_value(x);
    children_[0]->update_pattern(y);
    const Vec adjoint = children_[0]->eval_vjp(y, w);
    children_[1]->update_pattern(x);
    return children_[1]->eval_vjp(x, adjoint);
}
} // namespace litepgo::energy
