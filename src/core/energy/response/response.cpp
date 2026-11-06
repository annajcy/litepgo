#include "litepgo/energy/response/response.hpp"
#include "litepgo/common/busy_scope.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <stdexcept>

namespace litepgo::energy {
namespace {
SpMat checked(const SpMat& p) {
    SpMat result = linalg::pattern_of(p);
    if (!linalg::is_canonical(result))
        throw std::invalid_argument("Response: invalid sparse pattern");
    return result;
}
SpMat checked_hessian(const SpMat& p, int n) {
    if (p.rows() != n || p.cols() != n)
        throw std::invalid_argument("Response: invalid Hessian dimensions");
    SpMat result = checked(p);
    if (!linalg::same_pattern(result, SpMat(result.transpose())))
        throw std::invalid_argument("Response: Hessian pattern must be symmetric");
    return result;
}
} // namespace
Response::Response(const SpMat& jacobian_pattern, bool fixed_pattern, std::optional<SpMat> hessian_pattern)
    : jacobian_pattern_(checked(jacobian_pattern)), fixed_pattern_(fixed_pattern) {
    if (hessian_pattern)
        hessian_pattern_ = checked_hessian(*hessian_pattern, num_inputs());
}
const SpMat& Response::hessian_pattern() const {
    if (!has_hessian())
        throw std::logic_error("Response does not support Hessians");
    return *hessian_pattern_;
}
SpMat Response::jacobian_pattern_at(const ConstVecRef&) const {
    return jacobian_pattern_;
}
SpMat Response::hessian_pattern_at(const ConstVecRef&) const {
    return hessian_pattern();
}
void Response::eval_hessian_impl(const ConstVecRef&, const ConstVecRef&, SpMat&) const {
    throw std::logic_error("Response does not implement Hessian evaluation");
}
void Response::check_input(const ConstVecRef& x) const {
    if (x.size() != num_inputs() || !x.allFinite())
        throw std::invalid_argument("Response: invalid input");
}
void Response::check_weights(const ConstVecRef& weights) const {
    if (weights.size() != num_outputs() || !weights.allFinite())
        throw std::invalid_argument("Response: invalid output weights");
}
bool Response::prepare_patterns(const ConstVecRef& x, bool jacobian, bool hessian) {
    if (fixed_pattern_)
        return false;
    std::optional<SpMat> next, next_hessian;
    if (jacobian) {
        next = jacobian_pattern_at(x);
        if (next->rows() != num_outputs() || next->cols() != num_inputs())
            throw std::logic_error("Response: pattern dimensions changed");
        *next = checked(*next);
    }
    if (hessian && has_hessian())
        next_hessian = checked_hessian(hessian_pattern_at(x), num_inputs());
    // Validate both before installing either pattern.
    bool changed = false;
    if (next && !linalg::same_pattern(*next, jacobian_pattern_)) {
        jacobian_pattern_ = std::move(*next);
        changed = true;
    }
    if (next_hessian && !linalg::same_pattern(*next_hessian, *hessian_pattern_)) {
        hessian_pattern_ = std::move(*next_hessian);
        changed = true;
    }
    return changed;
}
bool Response::update_patterns(const ConstVecRef& x) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    return prepare_patterns(x, true, true);
}
Vec Response::evaluate_checked(const ConstVecRef& x, SpMat* J) const {
    Vec value = eval_impl(x, J);
    if (value.size() != num_outputs() || !value.allFinite())
        throw std::logic_error("Response: invalid value");
    if (J && (!linalg::same_pattern(*J, jacobian_pattern_) || !J->coeffs().allFinite()))
        throw std::logic_error("Response: invalid Jacobian");
    return value;
}
Vec Response::eval_value(const ConstVecRef& x) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    return evaluate_checked(x, nullptr);
}
std::pair<Vec, SpMat> Response::eval_value_and_jacobian(const ConstVecRef& x) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    prepare_patterns(x, true, false);
    SpMat J = jacobian_pattern_;
    Vec value = evaluate_checked(x, &J);
    return {std::move(value), std::move(J)};
}
SpMat Response::eval_jacobian(const ConstVecRef& x) {
    return eval_value_and_jacobian(x).second;
}
Vec Response::eval_value_and_jacobian_into(const ConstVecRef& x, SpMat& J) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    if (!linalg::same_pattern(J, jacobian_pattern_))
        throw std::invalid_argument("Response: mismatched Jacobian pattern");
    if (!fixed_pattern_ && !linalg::same_pattern(checked(jacobian_pattern_at(x)), jacobian_pattern_))
        throw std::logic_error("Response: call update_patterns(x) before *_into");
    return evaluate_checked(x, &J);
}
void Response::eval_jacobian_into(const ConstVecRef& x, SpMat& J) {
    eval_value_and_jacobian_into(x, J);
}
void Response::evaluate_hessian_checked(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) const {
    eval_hessian_impl(x, weights, H);
    if (!linalg::same_pattern(H, hessian_pattern()) || !H.coeffs().allFinite() ||
        !H.isApprox(SpMat(H.transpose()), 1e-12))
        throw std::logic_error("Response: invalid Hessian");
}
SpMat Response::eval_hessian(const ConstVecRef& x, const ConstVecRef& weights) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    check_weights(weights);
    hessian_pattern(); // Reject unsupported requests before invoking pattern hooks.
    prepare_patterns(x, false, true);
    SpMat H = hessian_pattern();
    evaluate_hessian_checked(x, weights, H);
    return H;
}
void Response::eval_hessian_into(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    check_weights(weights);
    if (!linalg::same_pattern(H, hessian_pattern()))
        throw std::invalid_argument("Response: mismatched Hessian pattern");
    if (!fixed_pattern_ &&
        !linalg::same_pattern(checked_hessian(hessian_pattern_at(x), num_inputs()), hessian_pattern()))
        throw std::logic_error("Response: call update_patterns(x) before *_into");
    evaluate_hessian_checked(x, weights, H);
}
Vec Response::eval_jvp(const ConstVecRef& x, const ConstVecRef& dx) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    if (dx.size() != num_inputs() || !dx.allFinite())
        throw std::invalid_argument("Response: invalid jvp direction");
    Vec result = eval_jvp_impl(x, dx);
    if (result.size() != num_outputs() || !result.allFinite())
        throw std::logic_error("Response: invalid jvp result");
    return result;
}
Vec Response::eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& dx) {
    check_input(x);
    if (dx.size() != num_inputs() || !dx.allFinite())
        throw std::invalid_argument("Response: invalid jvp direction");
    prepare_patterns(x, true, false);
    SpMat J = jacobian_pattern_;
    evaluate_checked(x, &J);
    return J * dx;
}
Vec Response::eval_vjp(const ConstVecRef& x, const ConstVecRef& weights) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    check_weights(weights);
    Vec result = eval_vjp_impl(x, weights);
    if (result.size() != num_inputs() || !result.allFinite())
        throw std::logic_error("Response: invalid vjp result");
    return result;
}
Vec Response::eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& weights) {
    check_input(x);
    check_weights(weights);
    prepare_patterns(x, true, false);
    SpMat J = jacobian_pattern_;
    evaluate_checked(x, &J);
    return J.transpose() * weights;
}
} // namespace litepgo::energy
