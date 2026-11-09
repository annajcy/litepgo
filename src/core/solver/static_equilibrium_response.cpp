#include "litepgo/solver/static_equilibrium_response.hpp"
#include "litepgo/dof/reduced_matrix.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>
namespace litepgo::solver {
StaticEquilibriumResponse::StaticEquilibriumResponse(int num_parameters, dof::DofReduction fixed, Vec initial,
                                                     NewtonOptions options, MakeEnergy make_energy,
                                                     MakeResponse make_response)
    : parameters_(num_parameters), fixed_(std::move(fixed)), initial_(std::move(initial)), options_(options),
      make_energy_(std::move(make_energy)), make_response_(std::move(make_response)) {
    if (num_parameters <= 0 || initial_.size() != fixed_.num_dofs() || !initial_.allFinite() || !make_energy_ ||
        !make_response_ || !std::isfinite(options_.gradient_tolerance) || options_.gradient_tolerance < 0 ||
        options_.max_iterations < 0 || options_.max_line_search_steps < 1 ||
        !(options_.line_search_shrink > 0 && options_.line_search_shrink < 1))
        throw std::invalid_argument("StaticEquilibriumResponse: invalid initial state, options or callbacks");
}

namespace {
class EquilibriumEvaluation final : public energy::ResponseEvaluation {
  public:
    EquilibriumEvaluation(Vec p, dof::DofReduction fixed, Vec initial, NewtonOptions options,
                          StaticEquilibriumResponse::MakeEnergy make_energy,
                          StaticEquilibriumResponse::MakeResponse make_response)
        : ResponseEvaluation(p.size(), initial.size()), parameters_(std::move(p)), fixed_(std::move(fixed)),
          initial_(std::move(initial)), options_(options), make_energy_(std::move(make_energy)),
          make_response_(std::move(make_response)) {}

  private:
    Vec value_impl() override {
        auto potential = make_energy_(parameters_);
        if (!potential || potential->num_dofs() != num_outputs())
            throw std::invalid_argument("StaticEquilibriumResponse: incompatible energy");
        SpMat final_hessian;
        const auto result = newton_solve(*potential, fixed_, initial_, options_, solver_, {}, &final_hessian);
        if (result.status != NewtonStatus::Converged)
            throw std::domain_error("StaticEquilibriumResponse: equilibrium did not converge");
        potential_ = std::move(potential);
        final_hessian_ = std::move(final_hessian);
        return result.x;
    }
    void prepare() {
        const Vec& u = value();
        if (prepared_ || fixed_.num_free() == 0)
            return;
        auto force = make_response_(u);
        if (!force || force->num_inputs() != num_inputs() || force->num_outputs() != num_outputs())
            throw std::invalid_argument("StaticEquilibriumResponse: incompatible force response");
        const SpMat exact = options_.hessian == energy::HessianKind::Exact
                                ? final_hessian_
                                : potential_->eval_hessian(u, energy::HessianKind::Exact);
        const dof::ReducedMatrix reduced(fixed_, exact), previous(fixed_, final_hessian_);
        K_ = reduced.matrix();
        const bool same = linalg::same_pattern(K_, previous.matrix());
        // Retry safely after failures: numeric factorization may have changed solver state.
        if ((!same && !solver_.analyze(K_)) || !solver_.factorize(K_))
            throw std::domain_error("StaticEquilibriumResponse: singular equilibrium stiffness");
        force_ = force->evaluate(parameters_);
        force_owner_ = std::move(force);
        prepared_ = true;
    }
    Vec solve(const Vec& rhs) {
        if (!rhs.allFinite())
            throw std::domain_error("StaticEquilibriumResponse: nonfinite sensitivity right-hand side");
        const Vec solution = solver_.solve(rhs);
        if (!solution.allFinite() || (K_ * solution - rhs).norm() > 1e-8 * (1 + rhs.norm()))
            throw std::domain_error("StaticEquilibriumResponse: inaccurate sensitivity solve");
        return solution;
    }
    Vec jvp_impl(const ConstVecRef& v) override {
        prepare();
        Vec full = Vec::Zero(num_outputs());
        if (fixed_.num_free())
            fixed_.expand(solve(-fixed_.reduce(force_->jvp(v))), full);
        return full;
    }
    Vec vjp_impl(const ConstVecRef& w) override {
        prepare();
        if (!fixed_.num_free())
            return Vec::Zero(num_inputs());
        Vec full = Vec::Zero(num_outputs());
        fixed_.expand(solve(fixed_.reduce(w)), full);
        return -force_->vjp(full);
    }
    SpMat jacobian_impl() override {
        prepare();
        SpMat J(num_outputs(), num_inputs());
        if (!fixed_.num_free())
            return J;
        const SpMat B = force_->jacobian();
        for (int c = 0; c < num_inputs(); ++c) {
            const Vec sensitivity = solve(-fixed_.reduce(Vec(B.col(c))));
            for (int r : fixed_.free_dofs())
                J.insert(r, c) = sensitivity[fixed_.free_index(r)];
        }
        J.makeCompressed();
        return J;
    }
    Vec parameters_;
    dof::DofReduction fixed_;
    Vec initial_;
    NewtonOptions options_;
    StaticEquilibriumResponse::MakeEnergy make_energy_;
    StaticEquilibriumResponse::MakeResponse make_response_;
    std::shared_ptr<energy::Energy> potential_;
    std::shared_ptr<energy::Response> force_owner_;
    std::shared_ptr<energy::ResponseEvaluation> force_;
    linalg::AutoIndefiniteSparseSolver solver_;
    SpMat final_hessian_, K_;
    bool prepared_ = false;
};
} // namespace
std::shared_ptr<energy::ResponseEvaluation> StaticEquilibriumResponse::make_evaluation(const ConstVecRef& p) {
    return std::make_shared<EquilibriumEvaluation>(p, fixed_, initial_, options_, make_energy_, make_response_);
}
} // namespace litepgo::solver
