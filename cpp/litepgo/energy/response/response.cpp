#include "litepgo/energy/response/response.hpp"
#include "litepgo/core/busy_scope.hpp"
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
    BusyScope scope(busy_, "Response");
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
    BusyScope scope(busy_, "Response");
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
} // namespace litepgo::energy
