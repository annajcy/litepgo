#include "litepgo/energy/response/composed_response.hpp"
#include <stdexcept>
namespace litepgo::energy {
ComposedResponse::ComposedResponse(std::shared_ptr<Response> outer, std::shared_ptr<Response> inner)
    : outer_(std::move(outer)), inner_(std::move(inner)) {
    if (!outer_ || !inner_)
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    if (outer_->num_inputs() != inner_->num_outputs())
        throw std::invalid_argument("ComposedResponse: incompatible responses");
    num_inputs_ = inner_->num_inputs();
    num_outputs_ = outer_->num_outputs();
}
bool ComposedResponse::has_hessian() const {
    return outer_->has_hessian() && inner_->has_hessian();
}
namespace {
class ComposedEvaluation final : public ResponseEvaluation {
  public:
    ComposedEvaluation(std::shared_ptr<Response> outer, std::shared_ptr<ResponseEvaluation> inner)
        : ResponseEvaluation(inner->num_inputs(), outer->num_outputs()), outer_(std::move(outer)),
          inner_(std::move(inner)) {}
    bool has_hessian() const override {
        return outer_->has_hessian() && inner_->has_hessian();
    }

  private:
    ResponseEvaluation& outer_state() {
        if (!outer_state_)
            outer_state_ = outer_->evaluate(inner_->value());
        return *outer_state_;
    }
    Vec value_impl() override {
        return outer_state().value();
    }
    SpMat jacobian_impl() override {
        const SpMat inner_J = inner_->jacobian();
        return outer_state().jacobian() * inner_J;
    }
    Vec jvp_impl(const ConstVecRef& v) override {
        auto& outer = outer_state();
        return outer.jvp(inner_->jvp(v));
    }
    Vec vjp_impl(const ConstVecRef& w) override {
        return inner_->vjp(outer_state().vjp(w));
    }
    SpMat hessian_impl(const ConstVecRef& w) override {
        const SpMat J = inner_->jacobian();
        auto& outer = outer_state();
        return SpMat(J.transpose() * outer.hessian(w) * J + inner_->hessian(outer.vjp(w)));
    }
    std::shared_ptr<Response> outer_;
    std::shared_ptr<ResponseEvaluation> inner_, outer_state_;
};
} // namespace
std::shared_ptr<ResponseEvaluation> ComposedResponse::make_evaluation(const ConstVecRef& x) {
    return std::make_shared<ComposedEvaluation>(outer_, inner_->evaluate(x));
}
} // namespace litepgo::energy
