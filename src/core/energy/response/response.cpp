#include "litepgo/energy/response/response.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <stdexcept>
namespace litepgo::energy {
void Response::check_input(const ConstVecRef& x) const {
    if (x.size() != num_inputs() || !x.allFinite())
        throw std::invalid_argument("Response: invalid input");
}
void Response::check_weights(const ConstVecRef& w) const {
    if (w.size() != num_outputs() || !w.allFinite())
        throw std::invalid_argument("Response: invalid output weights");
}
void Response::check_value(const Vec& v) const {
    if (v.size() != num_outputs() || !v.allFinite())
        throw std::logic_error("Response: invalid value");
}
Vec Response::eval_value(const ConstVecRef& x) {
    check_input(x);
    Vec value = eval_value_impl(x);
    check_value(value);
    return value;
}
std::pair<Vec, SpMat> Response::eval_value_and_jacobian(const ConstVecRef& x) {
    check_input(x);
    auto result = eval_value_and_jacobian_impl(x);
    check_value(result.first);
    const auto& J = result.second;
    if (J.rows() != num_outputs() || J.cols() != num_inputs())
        throw std::logic_error("Response: invalid Jacobian");
#ifndef NDEBUG
    if (!linalg::is_canonical(J))
        throw std::logic_error("Response: invalid Jacobian structure");
#endif
    if (!J.coeffs().allFinite())
        throw std::logic_error("Response: invalid Jacobian");
    return result;
}
SpMat Response::eval_jacobian(const ConstVecRef& x) {
    return eval_value_and_jacobian(x).second;
}
SpMat Response::eval_weighted_hessian_impl(const ConstVecRef&, const ConstVecRef&) {
    throw std::logic_error("Response does not implement Hessian evaluation");
}
SpMat Response::eval_hessian(const ConstVecRef& x, const ConstVecRef& weights) {
    check_input(x);
    check_weights(weights);
    if (!has_hessian())
        throw std::logic_error("Response does not support Hessians");
    SpMat H = eval_weighted_hessian_impl(x, weights);
    if (H.rows() != num_inputs() || H.cols() != num_inputs())
        throw std::logic_error("Response: invalid Hessian");
#ifndef NDEBUG
    if (!linalg::is_canonical(H))
        throw std::logic_error("Response: invalid Hessian structure");
#endif
    if (!H.coeffs().allFinite())
        throw std::logic_error("Response: invalid Hessian");
#ifndef NDEBUG
    if (!linalg::same_pattern(H, SpMat(H.transpose())) || (H.rows() != 0 && !H.isApprox(SpMat(H.transpose()), 1e-12)))
        throw std::logic_error("Response: invalid Hessian structure or symmetry");
#endif
    return H;
}
Vec Response::eval_jvp(const ConstVecRef& x, const ConstVecRef& dx) {
    check_input(x);
    if (dx.size() != num_inputs() || !dx.allFinite())
        throw std::invalid_argument("Response: invalid jvp direction");
    Vec result = eval_jvp_impl(x, dx);
    if (result.size() != num_outputs() || !result.allFinite())
        throw std::logic_error("Response: invalid jvp result");
    return result;
}
Vec Response::eval_vjp(const ConstVecRef& x, const ConstVecRef& weights) {
    check_input(x);
    check_weights(weights);
    Vec result = eval_vjp_impl(x, weights);
    if (result.size() != num_inputs() || !result.allFinite())
        throw std::logic_error("Response: invalid vjp result");
    return result;
}
Vec Response::eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& dx) {
    if (dx.size() != num_inputs() || !dx.allFinite())
        throw std::invalid_argument("Response: invalid jvp direction");
    return eval_jacobian(x) * dx;
}
Vec Response::eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& weights) {
    check_weights(weights);
    return eval_jacobian(x).transpose() * weights;
}
} // namespace litepgo::energy
