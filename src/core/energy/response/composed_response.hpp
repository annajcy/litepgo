#pragma once
#include "litepgo/energy/response/response.hpp"
#include <array>
#include <memory>

namespace litepgo::energy {
namespace binding {
template <class T> struct ResponseOwnerGC;
}
// outer(inner(x)); owns both responses. Ordinary evaluation prepares the composed pattern automatically.
class ComposedResponse final : public Response {
  public:
    ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner);

  private:
    template <class T> friend struct binding::ResponseOwnerGC;
    void check_children() const;
    SpMat jacobian_pattern_at(const ConstVecRef& x) const override;
    SpMat hessian_pattern_at(const ConstVecRef& x) const override;
    void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) const override;
    Vec eval_impl(const ConstVecRef& x, SpMat* J) const override;
    Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& v) override;
    Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& w) override;
    std::array<std::shared_ptr<Response>, 2> children_;
};
} // namespace litepgo::energy
