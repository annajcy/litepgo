#pragma once
#include "litepgo/energy/response/response.hpp"
#include <array>
#include <memory>

namespace litepgo::energy {
namespace binding {
template <class T> struct ResponseOwnerGC;
}
// outer(inner(x)); owns both responses. Update the outermost pattern before derivatives.
class ComposedResponse final : public Response {
  public:
    ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner);

  private:
    template <class T> friend struct binding::ResponseOwnerGC;
    void check_children() const;
    std::optional<SpMat> refresh_pattern(const ConstVecRef& x) override;
    std::optional<SpMat> refresh_hessian_pattern(const ConstVecRef& x) override;
    void eval_hessian_values(const ConstVecRef& x, const ConstVecRef& weights, SpMat& H) const override;
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override;
    Vec eval_jvp_values(const ConstVecRef& x, const ConstVecRef& v) const override;
    Vec eval_vjp_values(const ConstVecRef& x, const ConstVecRef& w) const override;
    std::array<std::shared_ptr<Response>, 2> children_;
};
} // namespace litepgo::energy
