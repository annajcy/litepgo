#pragma once
#include "litepgo/common/types.hpp"
#include <memory>
#include <optional>
#include <utility>

namespace litepgo::energy {
class Response;
// One input point. Values and derivatives are prepared on demand. Not concurrently callable.
class ResponseEvaluation {
  public:
    virtual ~ResponseEvaluation() = default;
    const Vec& value();
    SpMat jacobian();
    SpMat hessian(const ConstVecRef& weights);
    Vec jvp(const ConstVecRef& direction);
    Vec vjp(const ConstVecRef& weights);
    int num_inputs() const noexcept {
        return inputs_;
    }
    int num_outputs() const noexcept {
        return outputs_;
    }
    virtual bool has_hessian() const {
        return hessian_;
    }

  protected:
    ResponseEvaluation(int inputs, int outputs, bool hessian = false)
        : inputs_(inputs), outputs_(outputs), hessian_(hessian) {}
    virtual Vec value_impl() = 0;
    virtual SpMat jacobian_impl() = 0;
    virtual SpMat hessian_impl(const ConstVecRef& weights);
    virtual Vec jvp_impl(const ConstVecRef& direction);
    virtual Vec vjp_impl(const ConstVecRef& weights);
    void save_value(Vec value);

  private:
    friend class Response;
    int inputs_, outputs_;
    bool hessian_;
    std::optional<Vec> value_;
};

// Evaluations borrow their response; the response must outlive its evaluations.
// Inputs are copied; the response and any external callback state must remain unchanged while in use.
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
    std::shared_ptr<ResponseEvaluation> evaluate(const ConstVecRef& x);

  protected:
    Response() = default;
    virtual std::shared_ptr<ResponseEvaluation> make_evaluation(const ConstVecRef& x);
    virtual Vec value_at(const ConstVecRef& x);
    virtual std::pair<Vec, SpMat> value_jacobian_at(const ConstVecRef& x);
    virtual SpMat hessian_at(const ConstVecRef& x, const ConstVecRef& weights);
    virtual Vec jvp_at(const ConstVecRef& x, const ConstVecRef& direction);
    virtual Vec vjp_at(const ConstVecRef& x, const ConstVecRef& weights);

  private:
    friend class PointEvaluation;
};
} // namespace litepgo::energy
