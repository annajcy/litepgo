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
Response::Response(const SpMat& pattern, bool fixed, std::optional<SpMat> hessian_pattern)
    : pattern_(checked(pattern)), fixed_(fixed) {
    if (hessian_pattern)
        hessian_pattern_ = checked_hessian(*hessian_pattern, num_inputs());
}
const SpMat& Response::hessian_pattern() const {
    if (!has_hessian())
        throw std::logic_error("Response does not support Hessians");
    return *hessian_pattern_;
}
std::optional<SpMat> Response::refresh_hessian_pattern(const ConstVecRef&) {
    return std::nullopt;
}
void Response::eval_hessian_values(const ConstVecRef&, const ConstVecRef&, SpMat&) const {
    throw std::logic_error("Response does not implement Hessian evaluation");
}
SpMat Response::eval_hessian(const ConstVecRef& x, const ConstVecRef& weights) const {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    if (weights.size() != num_outputs() || !weights.allFinite())
        throw std::invalid_argument("Response: invalid Hessian weights");
    SpMat H = hessian_pattern();
    eval_hessian_values(x, weights, H);
    if (!linalg::same_pattern(H, hessian_pattern()) || !H.coeffs().allFinite() ||
        !H.isApprox(SpMat(H.transpose()), 1e-12))
        throw std::logic_error("Response: invalid Hessian");
    return H;
}
void Response::check_input(const ConstVecRef& x) const {
    if (x.size() != num_inputs() || !x.allFinite())
        throw std::invalid_argument("Response: invalid input");
}
bool Response::update_pattern(const ConstVecRef& x) {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    if (fixed_)
        return false;
    auto next = refresh_pattern(x);
    auto next_hessian = has_hessian() ? refresh_hessian_pattern(x) : std::nullopt;
    // Validate both before installing either pattern.
    if (next) {
        if (next->rows() != num_outputs() || next->cols() != num_inputs())
            throw std::logic_error("Response: pattern dimensions changed");
        *next = checked(*next);
    }
    if (next_hessian)
        *next_hessian = checked_hessian(*next_hessian, num_inputs());
    bool changed = false;
    if (next && !linalg::same_pattern(*next, pattern_)) {
        pattern_ = std::move(*next);
        changed = true;
    }
    if (next_hessian && !linalg::same_pattern(*next_hessian, *hessian_pattern_)) {
        hessian_pattern_ = std::move(*next_hessian);
        changed = true;
    }
    return changed;
}
Vec Response::eval(const ConstVecRef& x, SpMat* J) const {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    if (J && !linalg::same_pattern(*J, pattern_))
        throw std::invalid_argument("Response: mismatched Jacobian pattern");
    Vec value = eval_values(x, J);
    if (value.size() != num_outputs() || !value.allFinite())
        throw std::logic_error("Response: invalid value");
    if (J && (!linalg::same_pattern(*J, pattern_) || !J->coeffs().allFinite()))
        throw std::logic_error("Response: invalid Jacobian");
    return value;
}
SpMat Response::eval_jacobian(const ConstVecRef& x) const {
    SpMat J = pattern_;
    eval(x, &J);
    return J;
}
Vec Response::eval_jvp(const ConstVecRef& x, const ConstVecRef& dx) const {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    if (dx.size() != num_inputs() || !dx.allFinite())
        throw std::invalid_argument("Response: invalid jvp direction");
    Vec result = eval_jvp_values(x, dx);
    if (result.size() != num_outputs() || !result.allFinite())
        throw std::logic_error("Response: invalid jvp result");
    return result;
}
Vec Response::eval_jvp_values(const ConstVecRef& x, const ConstVecRef& dx) const {
    check_input(x);
    if (dx.size() != num_inputs() || !dx.allFinite())
        throw std::invalid_argument("Response: invalid jvp direction");
    SpMat J = pattern_;
    const Vec value = eval_values(x, &J);
    if (value.size() != num_outputs() || !value.allFinite() || !linalg::same_pattern(J, pattern_) ||
        !J.coeffs().allFinite())
        throw std::logic_error("Response: invalid Jacobian evaluation");
    return J * dx;
}
Vec Response::eval_vjp(const ConstVecRef& x, const ConstVecRef& gy) const {
    ::litepgo::detail::BusyScope scope(busy_, "Response");
    check_input(x);
    if (gy.size() != num_outputs() || !gy.allFinite())
        throw std::invalid_argument("Response: invalid vjp direction");
    Vec result = eval_vjp_values(x, gy);
    if (result.size() != num_inputs() || !result.allFinite())
        throw std::logic_error("Response: invalid vjp result");
    return result;
}
Vec Response::eval_vjp_values(const ConstVecRef& x, const ConstVecRef& gy) const {
    check_input(x);
    if (gy.size() != num_outputs() || !gy.allFinite())
        throw std::invalid_argument("Response: invalid vjp direction");
    SpMat J = pattern_;
    const Vec value = eval_values(x, &J);
    if (value.size() != num_outputs() || !value.allFinite() || !linalg::same_pattern(J, pattern_) ||
        !J.coeffs().allFinite())
        throw std::logic_error("Response: invalid Jacobian evaluation");
    return J.transpose() * gy;
}
} // namespace litepgo::energy
