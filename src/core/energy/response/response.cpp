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
} // namespace
Response::Response(const SpMat& pattern, bool fixed) : pattern_(checked(pattern)), fixed_(fixed) {}
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
    if (!next)
        return false;
    if (next->rows() != num_outputs() || next->cols() != num_inputs())
        throw std::logic_error("Response: pattern dimensions changed");
    SpMat pattern = checked(*next);
    if (linalg::same_pattern(pattern, pattern_))
        return false;
    pattern_ = std::move(pattern);
    return true;
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
