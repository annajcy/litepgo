#pragma once
#include "litepgo/common/types.hpp"
#include <atomic>
#include <optional>
#include <utility>

namespace litepgo::energy {
// Vector-valued function with fixed input/output dimensions and optional weighted Hessians.
// Ordinary evaluation prepares required sparse patterns automatically. Native *_into calls
// require update_patterns(x) and matching buffers; they overwrite values without resizing.
// One call at a time per instance. Infeasible states throw domain_error.
class Response {
  public:
    virtual ~Response() = default;
    Response(const Response&) = delete;
    Response& operator=(const Response&) = delete;
    int num_inputs() const noexcept {
        return static_cast<int>(jacobian_pattern_.cols());
    }
    int num_outputs() const noexcept {
        return static_cast<int>(jacobian_pattern_.rows());
    }
    bool has_fixed_pattern() const noexcept {
        return fixed_pattern_;
    }
    bool has_hessian() const noexcept {
        return hessian_pattern_.has_value();
    }

    Vec eval_value(const ConstVecRef& x);
    SpMat eval_jacobian(const ConstVecRef& x);
    std::pair<Vec, SpMat> eval_value_and_jacobian(const ConstVecRef& x);
    Vec eval_jvp(const ConstVecRef& x, const ConstVecRef& dx);
    Vec eval_vjp(const ConstVecRef& x, const ConstVecRef& weights);
    // Exact Hessian of weights.dot(R(x)), holding weights fixed. Full symmetric n-by-n matrix.
    SpMat eval_hessian(const ConstVecRef& x, const ConstVecRef& weights);

    // Returns true if either supported derivative pattern changes. Stores no current x or values.
    bool update_patterns(const ConstVecRef& x);
    const SpMat& jacobian_pattern() const {
        return jacobian_pattern_;
    }
    // Structural union of component Hessians, independent of weights; throws if unsupported.
    const SpMat& hessian_pattern() const;
    void eval_jacobian_into(const ConstVecRef& x, SpMat& jacobian);
    Vec eval_value_and_jacobian_into(const ConstVecRef& x, SpMat& jacobian);
    void eval_hessian_into(const ConstVecRef& x, const ConstVecRef& weights, SpMat& hessian);

  protected:
    // A fixed empty n-by-n Hessian pattern declares zero curvature; omission means unsupported.
    Response(const SpMat& jacobian_pattern, bool fixed_pattern, std::optional<SpMat> hessian_pattern = std::nullopt);
    // nullptr requests only the value. Otherwise overwrite all stored Jacobian values.
    virtual Vec eval_impl(const ConstVecRef& x, SpMat* jacobian) const = 0;
    // Defaults prepare and multiply an explicit Jacobian. Direct overrides need no pattern preparation.
    virtual Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& dx);
    virtual Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& weights);
    virtual void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights, SpMat& hessian) const;
    // Describe required structures without changing this object's stored patterns.
    virtual SpMat jacobian_pattern_at(const ConstVecRef& x) const;
    virtual SpMat hessian_pattern_at(const ConstVecRef& x) const;

  private:
    bool prepare_patterns(const ConstVecRef& x, bool jacobian, bool hessian);
    Vec evaluate_checked(const ConstVecRef& x, SpMat* jacobian) const;
    void evaluate_hessian_checked(const ConstVecRef& x, const ConstVecRef& weights, SpMat& hessian) const;
    void check_input(const ConstVecRef& x) const;
    void check_weights(const ConstVecRef& weights) const;
    SpMat jacobian_pattern_;
    std::optional<SpMat> hessian_pattern_;
    bool fixed_pattern_;
    std::atomic<bool> busy_ = false;
};
} // namespace litepgo::energy
