#include "litepgo/energy/response/response.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <stdexcept>
namespace litepgo::energy {
namespace {
void check_vector(const ConstVecRef& x, int size, const char* message) {
    if (x.size() != size || !x.allFinite())
        throw std::invalid_argument(message);
}
void check_matrix(const SpMat& matrix, int rows, int cols) {
    if (matrix.rows() != rows || matrix.cols() != cols || !matrix.coeffs().allFinite())
        throw std::logic_error("ResponseEvaluation: invalid derivative");
#ifndef NDEBUG
    if (!linalg::is_canonical(matrix))
        throw std::logic_error("ResponseEvaluation: invalid derivative structure");
#endif
}
} // namespace
void ResponseEvaluation::save_value(Vec v) {
    if (v.size() != outputs_ || !v.allFinite())
        throw std::logic_error("ResponseEvaluation: invalid value");
    value_ = std::move(v);
}
const Vec& ResponseEvaluation::value() {
    if (!value_)
        save_value(value_impl());
    return *value_;
}
SpMat ResponseEvaluation::jacobian() {
    SpMat J = jacobian_impl();
    check_matrix(J, outputs_, inputs_);
    return J;
}
SpMat ResponseEvaluation::hessian(const ConstVecRef& w) {
    check_vector(w, outputs_, "ResponseEvaluation: invalid output weights");
    if (!has_hessian())
        throw std::logic_error("Response does not support Hessians");
    SpMat H = hessian_impl(w);
    check_matrix(H, inputs_, inputs_);
#ifndef NDEBUG
    if (!linalg::same_pattern(H, SpMat(H.transpose())) || (H.rows() && !H.isApprox(SpMat(H.transpose()), 1e-12)))
        throw std::logic_error("ResponseEvaluation: invalid Hessian symmetry");
#endif
    return H;
}
Vec ResponseEvaluation::jvp(const ConstVecRef& d) {
    check_vector(d, inputs_, "ResponseEvaluation: invalid jvp direction");
    Vec result = jvp_impl(d);
    if (result.size() != outputs_ || !result.allFinite())
        throw std::logic_error("ResponseEvaluation: invalid jvp result");
    return result;
}
Vec ResponseEvaluation::vjp(const ConstVecRef& w) {
    check_vector(w, outputs_, "ResponseEvaluation: invalid output weights");
    Vec result = vjp_impl(w);
    if (result.size() != inputs_ || !result.allFinite())
        throw std::logic_error("ResponseEvaluation: invalid vjp result");
    return result;
}
SpMat ResponseEvaluation::hessian_impl(const ConstVecRef&) {
    throw std::logic_error("Response does not implement Hessians");
}
Vec ResponseEvaluation::jvp_impl(const ConstVecRef& d) {
    return jacobian() * d;
}
Vec ResponseEvaluation::vjp_impl(const ConstVecRef& w) {
    return jacobian().transpose() * w;
}
class PointEvaluation final : public ResponseEvaluation {
  public:
    PointEvaluation(Response& response, const ConstVecRef& x)
        : ResponseEvaluation(response.num_inputs(), response.num_outputs()), response_(response), x_(x) {}
    bool has_hessian() const override {
        return response_.has_hessian();
    }

  private:
    Vec value_impl() override {
        return response_.value_at(x_);
    }
    SpMat jacobian_impl() override {
        auto [v, J] = response_.value_jacobian_at(x_);
        save_value(std::move(v));
        return J;
    }
    SpMat hessian_impl(const ConstVecRef& w) override {
        return response_.hessian_at(x_, w);
    }
    Vec jvp_impl(const ConstVecRef& d) override {
        return response_.jvp_at(x_, d);
    }
    Vec vjp_impl(const ConstVecRef& w) override {
        return response_.vjp_at(x_, w);
    }
    Response& response_;
    Vec x_;
};
std::shared_ptr<ResponseEvaluation> Response::evaluate(const ConstVecRef& x) {
    check_vector(x, num_inputs(), "Response: invalid input");
    auto state = make_evaluation(x);
    if (!state || state->num_inputs() != num_inputs() || state->num_outputs() != num_outputs())
        throw std::logic_error("Response: invalid evaluation");
    state->owner_ = weak_from_this().lock();
    return state;
}
std::shared_ptr<ResponseEvaluation> Response::make_evaluation(const ConstVecRef& x) {
    return std::make_shared<PointEvaluation>(*this, x);
}
Vec Response::value_at(const ConstVecRef&) {
    throw std::logic_error("Response: missing value implementation");
}
std::pair<Vec, SpMat> Response::value_jacobian_at(const ConstVecRef&) {
    throw std::logic_error("Response: missing Jacobian implementation");
}
SpMat Response::hessian_at(const ConstVecRef&, const ConstVecRef&) {
    throw std::logic_error("Response does not implement Hessians");
}
Vec Response::jvp_at(const ConstVecRef& x, const ConstVecRef& d) {
    check_vector(d, num_inputs(), "Response: invalid jvp direction");
    return evaluate(x)->jacobian() * d;
}
Vec Response::vjp_at(const ConstVecRef& x, const ConstVecRef& w) {
    check_vector(w, num_outputs(), "Response: invalid output weights");
    return evaluate(x)->jacobian().transpose() * w;
}
} // namespace litepgo::energy
