#pragma once
#include "litepgo/energy/response/response.hpp"
#include <memory>

namespace litepgo::energy {
// outer(inner(x)); owns both responses. Ordinary evaluation prepares the composed pattern automatically.
class ComposedResponse final : public Response {
  public:
    ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner);
    int num_inputs() const noexcept override {
        return num_inputs_;
    }
    int num_outputs() const noexcept override {
        return num_outputs_;
    }
    bool has_hessian() const override;

  private:
    Vec eval_value_impl(const ConstVecRef& x) override;
    std::pair<Vec, SpMat> eval_value_and_jacobian_impl(const ConstVecRef& x) override;
    SpMat eval_weighted_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights) override;
    Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& v) override;
    Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& w) override;
    std::shared_ptr<Response> outer_, inner_;
    int num_inputs_, num_outputs_;
};
} // namespace litepgo::energy
