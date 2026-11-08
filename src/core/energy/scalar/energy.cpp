#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
namespace litepgo::energy {
Energy::Energy(int num_dofs) : num_dofs_(num_dofs) {
    if (num_dofs < 0)
        throw std::invalid_argument("Energy requires a non-negative DOF count");
}
EnergyEvaluation Energy::eval(const ConstVecRef& x, bool gradient, std::optional<HessianKind> hessian) const {
    check_size(x, "eval");
    if (hessian && *hessian != HessianKind::Exact && *hessian != HessianKind::Projected)
        throw std::invalid_argument("Energy: invalid Hessian kind");
    auto result = eval_impl(x, gradient, hessian);
    if (!std::isfinite(result.value))
        throw std::logic_error("Energy: values must be finite");
    if (result.gradient.has_value() != gradient || result.hessian.has_value() != hessian.has_value())
        throw std::logic_error("Energy: requested outputs missing or unexpected outputs returned");
    if (result.gradient && (result.gradient->size() != num_dofs_ || !result.gradient->allFinite()))
        throw std::logic_error("Energy: invalid gradient");
    if (result.hessian) {
        const auto& H = *result.hessian;
        if (H.rows() != num_dofs_ || H.cols() != num_dofs_ || !H.coeffs().allFinite())
            throw std::logic_error("Energy: invalid Hessian");
#ifndef NDEBUG
        if (!linalg::is_canonical(H) || !linalg::same_pattern(H, SpMat(H.transpose())) ||
            (H.rows() && !H.isApprox(SpMat(H.transpose()), 1e-12)))
            throw std::logic_error("Energy: invalid Hessian structure or symmetry");
#endif
    }
    return result;
}
double Energy::max_feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
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
    return eval(x).value;
}
Vec Energy::eval_gradient(const ConstVecRef& x) const {
    return std::move(*eval(x, true).gradient);
}
SpMat Energy::eval_hessian(const ConstVecRef& x, HessianKind kind) const {
    return std::move(*eval(x, false, kind).hessian);
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
