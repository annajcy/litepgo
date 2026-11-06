#pragma once
#include "litepgo/common/types.hpp"
#include <utility>

namespace litepgo::energy {
// Mathematical map with fixed dimensions. No pattern cache or shared execution state.
// Implementations with shared workspaces are responsible for protecting them.
class Response {
  public:
    virtual ~Response() = default;
    Response(const Response&) = delete;
    Response& operator=(const Response&) = delete;
    virtual int num_inputs() const noexcept = 0;
    virtual int num_outputs() const noexcept = 0;
    virtual bool has_hessian() const {
        return false;
    }
    Vec eval_value(const ConstVecRef& x);
    SpMat eval_jacobian(const ConstVecRef& x);
    std::pair<Vec, SpMat> eval_value_and_jacobian(const ConstVecRef& x);
    Vec eval_jvp(const ConstVecRef& x, const ConstVecRef& dx);
    Vec eval_vjp(const ConstVecRef& x, const ConstVecRef& weights);
    // Exact full symmetric Hessian of weights.dot(R(x)), holding weights fixed.
    SpMat eval_hessian(const ConstVecRef& x, const ConstVecRef& weights);

  protected:
    Response() = default;
    virtual Vec eval_value_impl(const ConstVecRef& x) = 0;
    virtual std::pair<Vec, SpMat> eval_value_and_jacobian_impl(const ConstVecRef& x) = 0;
    virtual SpMat eval_weighted_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights);
    virtual Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& dx);
    virtual Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& weights);

  private:
    void check_input(const ConstVecRef& x) const;
    void check_weights(const ConstVecRef& weights) const;
    void check_value(const Vec& value) const;
};
} // namespace litepgo::energy
