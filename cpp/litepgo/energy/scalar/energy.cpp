#include "litepgo/energy/scalar/energy.hpp"

#include "litepgo/core/busy_scope.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace litepgo::energy {

namespace {

SpMat checked_pattern(int num_dofs, const SpMat& pattern) {
    if (pattern.rows() != num_dofs || pattern.cols() != num_dofs) {
        throw std::invalid_argument("Energy pattern must be num_dofs x num_dofs");
    }
    SpMat checked = linalg::pattern_of(pattern);
    if (!linalg::is_canonical(checked)) {
        throw std::invalid_argument("Energy pattern must have sorted row indices and no duplicate entries");
    }
    return checked;
}

} // namespace

Energy::Energy(int num_dofs, const SpMat& pattern, bool fixed_pattern)
    : num_dofs_(num_dofs), fixed_pattern_(fixed_pattern) {
    if (num_dofs_ < 0) {
        throw std::invalid_argument("Energy requires a non-negative DOF count");
    }
    pattern_ = checked_pattern(num_dofs_, pattern);
}

bool Energy::update_pattern(const ConstVecRef& x) {
    BusyScope scope(busy_, "Energy");
    check_size(x, "update_pattern");
    if (fixed_pattern_) {
        return false;
    }
    std::optional<SpMat> pattern = refresh_pattern(x);
    if (!pattern) {
        return false;
    }
    pattern_ = checked_pattern(num_dofs_, *pattern);
    return true;
}

double Energy::eval(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const {
    BusyScope scope(busy_, "Energy");
    check_size(x, "eval");
    if (hessian != nullptr && !linalg::same_pattern(*hessian, pattern_)) {
        throw std::invalid_argument("Energy::eval: the hessian does not have this energy's current sparsity "
                                    "pattern; copy hessian_pattern() again after update_pattern()");
    }
    if (gradient != nullptr) {
        gradient->resize(num_dofs_);
    }
    const double value = eval_values(x, gradient, hessian, kind);
    if (!std::isfinite(value)) {
        throw std::logic_error("Energy::eval: the energy returned " + std::to_string(value) +
                               "; values must be finite, with std::domain_error thrown at infeasible states");
    }
    if (gradient && (gradient->size() != num_dofs_ || !gradient->allFinite()))
        throw std::logic_error("Energy::eval: invalid gradient");
    if (hessian && (!linalg::same_pattern(*hessian, pattern_) || !hessian->coeffs().allFinite()))
        throw std::logic_error("Energy::eval: invalid Hessian");
    return value;
}

double Energy::max_feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    BusyScope scope(busy_, "Energy");
    check_size(x, "max_feasible_step");
    if (dx.size() != num_dofs_ || !dx.allFinite()) {
        throw std::invalid_argument("Energy::max_feasible_step: dx must have num_dofs finite entries");
    }
    const double alpha = feasible_step(x, dx);
    if (!(alpha >= 0.0 && alpha <= 1.0)) {
        throw std::logic_error("Energy::max_feasible_step: the energy returned " + std::to_string(alpha) +
                               ", outside [0, 1]");
    }
    return alpha;
}

double Energy::eval_value(const ConstVecRef& x) const {
    return eval(x, nullptr, nullptr);
}

Vec Energy::eval_gradient(const ConstVecRef& x) const {
    Vec gradient;
    eval(x, &gradient, nullptr);
    return gradient;
}

SpMat Energy::eval_hessian(const ConstVecRef& x, HessianKind kind) const {
    SpMat hessian = pattern_;
    eval(x, nullptr, &hessian, kind);
    return hessian;
}

double Energy::feasible_step(const ConstVecRef&, const ConstVecRef&) const {
    return 1.0;
}

void Energy::check_size(const ConstVecRef& x, const char* caller) const {
    if (x.size() != num_dofs_ || !x.allFinite()) {
        throw std::invalid_argument(std::string("Energy::") + caller + ": x must have num_dofs finite entries");
    }
}

} // namespace litepgo::energy
