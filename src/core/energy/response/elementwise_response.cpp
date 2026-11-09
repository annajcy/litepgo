#include "litepgo/energy/response/elementwise_response.hpp"
namespace litepgo::energy {
namespace {
Vec gather(const dof::ResponseAssembler& assembler, int e, const ConstVecRef& x) {
    const auto ids = assembler.input_indices(e);
    Vec q(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i)
        q[i] = x[ids[i]];
    return q;
}
class ElementwiseEvaluation final : public ResponseEvaluation {
  public:
    ElementwiseEvaluation(const dof::ResponseAssembler& assembler, Response& transform, const ConstVecRef& x)
        : ResponseEvaluation(assembler.jacobian_pattern().cols(), assembler.jacobian_pattern().rows()),
          assembler_(assembler) {
        children_.reserve(assembler.num_elements());
        for (int e = 0; e < assembler.num_elements(); ++e)
            children_.push_back(transform.evaluate(gather(assembler, e, x)));
    }

  private:
    Vec assemble(SpMat* J) {
        const int count = assembler_.num_elements();
        std::vector<Vec> values(count);
        std::vector<Eigen::MatrixXd> jacobians(J ? count : 0);
        std::vector<dof::ResponseAssembler::ElementView> views;
        views.reserve(count);
        // Child calls stay serial because the shared transform may own workspaces.
        for (int e = 0; e < count; ++e) {
            if (J)
                jacobians[e] = children_[e]->jacobian();
            values[e] = children_[e]->value();
            if (J)
                views.emplace_back(values[e], jacobians[e]);
            else
                views.emplace_back(values[e]);
        }
        Vec result(num_outputs());
        assembler_.assemble(views, result, J);
        return result;
    }
    Vec value_impl() override {
        return assemble(nullptr);
    }
    SpMat jacobian_impl() override {
        SpMat J = assembler_.jacobian_pattern();
        save_value(assemble(&J));
        return J;
    }
    Vec jvp_impl(const ConstVecRef& v) override {
        Vec result(num_outputs());
        for (int e = 0; e < assembler_.num_elements(); ++e) {
            const int width = children_[e]->num_outputs();
            result.segment(e * width, width) = children_[e]->jvp(gather(assembler_, e, v));
        }
        return result;
    }
    Vec vjp_impl(const ConstVecRef& w) override {
        Vec result = Vec::Zero(num_inputs());
        for (int e = 0; e < assembler_.num_elements(); ++e) {
            const int width = children_[e]->num_outputs();
            const Vec product = children_[e]->vjp(w.segment(e * width, width));
            const auto ids = assembler_.input_indices(e);
            for (std::size_t i = 0; i < ids.size(); ++i)
                result[ids[i]] += product[i];
        }
        return result;
    }
    const dof::ResponseAssembler& assembler_;
    std::vector<std::shared_ptr<ResponseEvaluation>> children_;
};
} // namespace
ElementwiseResponse::ElementwiseResponse(dof::ResponseAssembler assembler, std::shared_ptr<Response> child)
    : assembler_(std::move(assembler)), transform_(std::move(child)) {}
std::shared_ptr<ResponseEvaluation> ElementwiseResponse::make_evaluation(const ConstVecRef& x) {
    return std::make_shared<ElementwiseEvaluation>(assembler_, *transform_, x);
}
} // namespace litepgo::energy
