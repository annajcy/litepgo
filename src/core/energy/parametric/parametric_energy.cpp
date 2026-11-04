#include "litepgo/energy/parametric/parametric_energy.hpp"
#include "litepgo/common/busy_scope.hpp"
#include "litepgo/energy/parametric/parametric_view.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <cmath>
#include <stdexcept>

namespace litepgo::energy {
namespace {
void check(const ConstVecRef& x, int n) {
    if (x.size() != n || !x.allFinite())
        throw std::invalid_argument("ParametricEnergy: invalid input");
}
SpMat pattern(const SpMat& input, int rows, int cols) {
    SpMat result = linalg::pattern_of(input);
    if (result.rows() != rows || result.cols() != cols || !linalg::is_canonical(result))
        throw std::invalid_argument("ParametricEnergy: invalid pattern");
    return result;
}
} // namespace
class ParametricEnergy::EnergyView final : public Energy, public detail::ParametricView {
  public:
    EnergyView(std::shared_ptr<ParametricEnergy> model, Vec p)
        : Energy(model->num_dofs(), model->state_pattern_, model->fixed_pattern_), ParametricView(std::move(model)),
          p_(std::move(p)) {}

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef& u) override {
        if (!model_)
            throw std::logic_error("ParametricEnergy: cleared view");
        ::litepgo::detail::BusyScope scope(model_->busy_, "ParametricEnergy");
        auto next = pattern(model_->state_pattern_at(u, p_), num_dofs(), num_dofs());
        if (linalg::same_pattern(next, hessian_pattern()))
            return std::nullopt;
        return next;
    }
    double eval_values(const ConstVecRef& u, Vec* g, SpMat* K, HessianKind kind) const override {
        if (!model_)
            throw std::logic_error("ParametricEnergy: cleared view");
        ::litepgo::detail::BusyScope scope(model_->busy_, "ParametricEnergy");
        return model_->eval_state_values(u, p_, g, K, kind);
    }
    double feasible_step(const ConstVecRef& u, const ConstVecRef& du) const override {
        if (!model_)
            throw std::logic_error("ParametricEnergy: cleared view");
        ::litepgo::detail::BusyScope scope(model_->busy_, "ParametricEnergy");
        return model_->eval_feasible_step(u, p_, du);
    }
    Vec p_;
};
class ParametricEnergy::ForceView final : public Response, public detail::ParametricView {
  public:
    ForceView(std::shared_ptr<ParametricEnergy> model, Vec u)
        : Response(model->parameter_pattern_, model->fixed_pattern_), ParametricView(std::move(model)),
          u_(std::move(u)) {}

  private:
    std::optional<SpMat> refresh_pattern(const ConstVecRef& p) override {
        if (!model_)
            throw std::logic_error("ParametricEnergy: cleared view");
        ::litepgo::detail::BusyScope scope(model_->busy_, "ParametricEnergy");
        return pattern(model_->parameter_pattern_at(u_, p), num_outputs(), num_inputs());
    }
    Vec eval_values(const ConstVecRef& p, SpMat* B) const override {
        if (!model_)
            throw std::logic_error("ParametricEnergy: cleared view");
        ::litepgo::detail::BusyScope scope(model_->busy_, "ParametricEnergy");
        Vec g(num_outputs());
        const double value = model_->eval_state_values(u_, p, &g, nullptr, HessianKind::Exact);
        if (!std::isfinite(value))
            throw std::logic_error("ParametricEnergy: nonfinite energy");
        if (B)
            model_->eval_parameter_jacobian(u_, p, *B);
        return g;
    }
    Vec eval_jvp_values(const ConstVecRef& p, const ConstVecRef& v) const override {
        if (!model_)
            throw std::logic_error("ParametricEnergy: cleared view");
        ::litepgo::detail::BusyScope scope(model_->busy_, "ParametricEnergy");
        return model_->eval_parameter_jvp(u_, p, v);
    }
    Vec eval_vjp_values(const ConstVecRef& p, const ConstVecRef& w) const override {
        if (!model_)
            throw std::logic_error("ParametricEnergy: cleared view");
        ::litepgo::detail::BusyScope scope(model_->busy_, "ParametricEnergy");
        return model_->eval_parameter_vjp(u_, p, w);
    }
    Vec u_;
};
ParametricEnergy::ParametricEnergy(const SpMat& K, const SpMat& B, bool fixed)
    : state_pattern_(pattern(K, static_cast<int>(K.rows()), static_cast<int>(K.rows()))),
      parameter_pattern_(pattern(B, static_cast<int>(K.rows()), static_cast<int>(B.cols()))), fixed_pattern_(fixed) {}
std::shared_ptr<Energy> ParametricEnergy::energy_at(const ConstVecRef& p) {
    check(p, num_parameters());
    return std::make_shared<EnergyView>(shared_from_this(), Vec(p));
}
std::shared_ptr<Response> ParametricEnergy::force_at(const ConstVecRef& u) {
    check(u, num_dofs());
    return std::make_shared<ForceView>(shared_from_this(), Vec(u));
}
SpMat ParametricEnergy::state_pattern_at(const ConstVecRef&, const ConstVecRef&) const {
    return state_pattern_;
}
SpMat ParametricEnergy::parameter_pattern_at(const ConstVecRef&, const ConstVecRef&) const {
    return parameter_pattern_;
}
double ParametricEnergy::eval_feasible_step(const ConstVecRef&, const ConstVecRef&, const ConstVecRef&) const {
    return 1.;
}
SpMat ParametricEnergy::parameter_jacobian(const ConstVecRef& u, const ConstVecRef& p) const {
    check(u, num_dofs());
    check(p, num_parameters());
    SpMat B = fixed_pattern_ ? parameter_pattern_ : pattern(parameter_pattern_at(u, p), num_dofs(), num_parameters());
    const SpMat expected = B;
    eval_parameter_jacobian(u, p, B);
    if (!linalg::same_pattern(B, expected) || !B.coeffs().allFinite())
        throw std::logic_error("ParametricEnergy: invalid parameter Jacobian");
    return B;
}
Vec ParametricEnergy::eval_parameter_jvp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& v) const {
    check(v, num_parameters());
    return parameter_jacobian(u, p) * v;
}
Vec ParametricEnergy::eval_parameter_vjp(const ConstVecRef& u, const ConstVecRef& p, const ConstVecRef& w) const {
    check(w, num_dofs());
    return parameter_jacobian(u, p).transpose() * w;
}
} // namespace litepgo::energy
