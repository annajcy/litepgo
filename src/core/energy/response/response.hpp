#pragma once
#include "litepgo/common/types.hpp"
#include <atomic>
#include <optional>

namespace litepgo::energy {
// Vector-valued function. Inputs/outputs have fixed dimensions; Jacobians may be rectangular.
// Before requesting derivatives, update_pattern(x), then copy the required pattern.
// has_fixed_pattern describes both supported derivative patterns.
// eval overwrites values only. Value-only calls need no pattern update.
// One call at a time per instance. Infeasible states throw domain_error.
class Response {
  public:
    virtual ~Response() = default;
    Response(const Response&) = delete;
    Response& operator=(const Response&) = delete;
    int num_inputs() const {
        return static_cast<int>(pattern_.cols());
    }
    int num_outputs() const {
        return static_cast<int>(pattern_.rows());
    }
    bool has_fixed_pattern() const {
        return fixed_;
    }
    const SpMat& jacobian_pattern() const {
        return pattern_;
    }
    bool has_hessian() const noexcept {
        return hessian_pattern_.has_value();
    }
    // Union of all component Hessian patterns, independent of output weights.
    // Throws when second derivatives are unsupported.
    const SpMat& hessian_pattern() const;
    // Exact Hessian of weights.dot(R(x)), holding weights fixed. Full symmetric n-by-n matrix.
    SpMat eval_hessian(const ConstVecRef& x, const ConstVecRef& weights) const;
    bool update_pattern(const ConstVecRef& x);
    Vec eval(const ConstVecRef& x, SpMat* jacobian = nullptr) const;
    Vec eval_value(const ConstVecRef& x) const {
        return eval(x);
    }
    SpMat eval_jacobian(const ConstVecRef& x) const;
    // Dense products; defaults evaluate the explicit Jacobian on each call.
    // Dynamic responses require update_pattern(x), as for eval_jacobian.
    Vec eval_jvp(const ConstVecRef& x, const ConstVecRef& dx) const;
    Vec eval_vjp(const ConstVecRef& x, const ConstVecRef& gy) const;

  protected:
    // An optional n-by-n Hessian pattern enables second derivatives; a fixed empty pattern declares zero curvature.
    Response(const SpMat& pattern, bool fixed, std::optional<SpMat> hessian_pattern = std::nullopt);
    virtual Vec eval_jvp_values(const ConstVecRef& x, const ConstVecRef& dx) const;
    virtual Vec eval_vjp_values(const ConstVecRef& x, const ConstVecRef& gy) const;
    // Overwrite every stored value without changing H's pattern.
    virtual void eval_hessian_values(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) const;
    // Called only for dynamic responses with Hessian support; nullopt keeps the current pattern.
    virtual std::optional<SpMat> refresh_hessian_pattern(const ConstVecRef& x);

  private:
    virtual std::optional<SpMat> refresh_pattern(const ConstVecRef& x) = 0;
    virtual Vec eval_values(const ConstVecRef& x, SpMat* jacobian) const = 0;
    void check_input(const ConstVecRef& x) const;
    SpMat pattern_;
    std::optional<SpMat> hessian_pattern_;
    bool fixed_;
    mutable std::atomic<bool> busy_ = false;
};
} // namespace litepgo::energy
