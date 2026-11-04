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
SpMat initial_pattern(const std::shared_ptr<Response>& outer, const std::shared_ptr<Response>& inner) {
    if (!outer || !inner || outer->num_inputs() != inner->num_outputs())
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    return product_pattern(outer->jacobian_pattern(), inner->jacobian_pattern());
}
} // namespace
ComposedResponse::ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner)
    : Response(initial_pattern(outer, inner),
               outer && inner && outer->has_fixed_pattern() && inner->has_fixed_pattern()),
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
