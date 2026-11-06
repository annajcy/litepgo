#include "litepgo/energy/response/elementwise_response.hpp"
namespace litepgo::energy {
ElementwiseResponse::ElementwiseResponse(dof::ResponseAssembler assembler, std::shared_ptr<Response> child)
    : FixedPatternResponse(assembler.jacobian_pattern()), assembler_(std::move(assembler)),
      children_{std::move(child)} {}
Response& ElementwiseResponse::transform() const {
    if (!children_[0])
        throw std::logic_error("ElementwiseResponse: cleared transform");
    return *children_[0];
}
Vec ElementwiseResponse::gather(int e, const ConstVecRef& x) const {
    const auto ids = assembler_.input_indices(e);
    Vec q(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i)
        q[i] = x[ids[i]];
    return q;
}
Vec ElementwiseResponse::eval_impl(const ConstVecRef& x, SpMat* J) const {
    auto& local = transform();
    const int count = assembler_.num_elements();
    std::vector<Vec> values(count);
    std::vector<Eigen::MatrixXd> jacobians(J ? count : 0);
    std::vector<dof::ResponseAssembler::ElementView> views;
    views.reserve(count);
    // shortcut: local Response calls are serial because instances reject concurrent evaluation; batch transforms if
    // profiling warrants it.
    for (int e = 0; e < count; ++e) {
        const Vec q = gather(e, x);
        if (J) {
            auto [value, derivative] = local.eval_value_and_jacobian(q);
            values[e] = std::move(value);
            jacobians[e] = derivative;
            views.emplace_back(values[e], jacobians[e]);
        } else {
            values[e] = local.eval_value(q);
            views.emplace_back(values[e]);
        }
    }
    Vec value(num_outputs());
    assembler_.assemble(views, value, J);
    return value;
}
Vec ElementwiseResponse::eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& v) {
    auto& local = transform();
    Vec result(num_outputs());
    for (int e = 0; e < assembler_.num_elements(); ++e) {
        const Vec q = gather(e, x);
        result.segment(e * local.num_outputs(), local.num_outputs()) = local.eval_jvp(q, gather(e, v));
    }
    return result;
}
Vec ElementwiseResponse::eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& w) {
    auto& local = transform();
    Vec result = Vec::Zero(num_inputs());
    for (int e = 0; e < assembler_.num_elements(); ++e) {
        const Vec q = gather(e, x);
        const Vec product = local.eval_vjp(q, w.segment(e * local.num_outputs(), local.num_outputs()));
        const auto ids = assembler_.input_indices(e);
        for (std::size_t i = 0; i < ids.size(); ++i)
            result[ids[i]] += product[i];
    }
    return result;
}
} // namespace litepgo::energy
