#include "litepgo/solver/static_equilibrium_response.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo::solver {
namespace {
SpMat state_pattern(int parameters, const dof::DofReduction& fixed) {
    if (parameters <= 0)
        throw std::invalid_argument("StaticEquilibriumResponse: positive parameter count required");
    // shortcut: the declared sensitivity pattern is dense on free DOFs; make patterns lazy for large parameter fields.
    SpMat pattern(fixed.num_dofs(), parameters);
    for (int c = 0; c < parameters; ++c)
        for (int r : fixed.free_dofs())
            pattern.insert(r, c) = 1;
    pattern.makeCompressed();
    return pattern;
}
} // namespace
StaticEquilibriumResponse::StaticEquilibriumResponse(int num_parameters, dof::DofReduction fixed, Vec initial,
                                                     NewtonOptions options, MakeEnergy make_energy,
                                                     MakeResponse make_response)
    : energy::FixedPatternResponse(state_pattern(num_parameters, fixed)), fixed_(std::move(fixed)),
      initial_(std::move(initial)), options_(options), make_energy_(std::move(make_energy)),
      make_response_(std::move(make_response)) {
    if (initial_.size() != fixed_.num_dofs() || !initial_.allFinite() || !make_energy_ || !make_response_ ||
        !std::isfinite(options_.gradient_tolerance) || options_.gradient_tolerance < 0 || options_.max_iterations < 0 ||
        options_.max_line_search_steps < 1 || !(options_.line_search_shrink > 0 && options_.line_search_shrink < 1))
        throw std::invalid_argument("StaticEquilibriumResponse: invalid initial state, options or callbacks");
}
Vec StaticEquilibriumResponse::eval_impl(const ConstVecRef& p, SpMat* J) const {
    return evaluate(p, J, nullptr, false);
}
Vec StaticEquilibriumResponse::eval_jvp_impl(const ConstVecRef& p, const ConstVecRef& v) {
    const Vec direction = v;
    return evaluate(p, nullptr, &direction, false);
}
Vec StaticEquilibriumResponse::eval_vjp_impl(const ConstVecRef& p, const ConstVecRef& w) {
    const Vec direction = w;
    return evaluate(p, nullptr, &direction, true);
}
Vec StaticEquilibriumResponse::evaluate(const ConstVecRef& p, SpMat* J, const Vec* direction, bool transpose) const {
    const Vec parameters = p;
    auto potential = make_energy_(parameters);
    if (!potential || potential->num_dofs() != num_outputs())
        throw std::invalid_argument("StaticEquilibriumResponse: incompatible energy");
    linalg::AutoIndefiniteSparseSolver linear_solver;
    const auto result = newton_solve(*potential, fixed_, initial_, options_, linear_solver);
    if (result.status != NewtonStatus::Converged)
        throw std::domain_error("StaticEquilibriumResponse: equilibrium did not converge");
    if (direction && fixed_.num_free() == 0)
        return Vec::Zero(transpose ? num_inputs() : num_outputs());
    if ((J || direction) && fixed_.num_free() > 0) {
        auto force = make_response_(result.x);
        if (!force || force->num_inputs() != num_inputs() || force->num_outputs() != num_outputs())
            throw std::invalid_argument("StaticEquilibriumResponse: incompatible force response");
        potential->update_pattern(result.x);
        const SpMat K = fixed_.reduce(potential->eval_hessian(result.x, energy::HessianKind::Exact));
        if (!linear_solver.compute(K))
            throw std::domain_error("StaticEquilibriumResponse: singular equilibrium stiffness");
        const auto solve = [&](const Vec& rhs) -> Vec {
            if (!rhs.allFinite())
                throw std::domain_error("StaticEquilibriumResponse: nonfinite sensitivity right-hand side");
            const Vec solution = linear_solver.solve(rhs);
            if (!solution.allFinite() || (K * solution - rhs).norm() > 1e-8 * (1 + rhs.norm()))
                throw std::domain_error("StaticEquilibriumResponse: inaccurate sensitivity solve");
            return solution;
        };
        if (direction) {
            // K is the symmetric exact energy Hessian, so its factorization also solves K^T.
            const Vec rhs =
                transpose ? fixed_.reduce(*direction) : -fixed_.reduce(force->eval_jvp(parameters, *direction));
            Vec full = Vec::Zero(num_outputs());
            fixed_.expand(solve(rhs), full);
            if (transpose)
                return -force->eval_vjp(parameters, full);
            return full;
        }
        const SpMat B = force->eval_jacobian(parameters);
        for (int c = 0; c < num_inputs(); ++c) {
            const Vec rhs = -fixed_.reduce(Vec(B.col(c)));
            const Vec sensitivity = solve(rhs);
            for (SpMat::InnerIterator it(*J, c); it; ++it)
                it.valueRef() = sensitivity[fixed_.free_index(static_cast<int>(it.row()))];
        }
    }
    return result.x;
}
} // namespace litepgo::solver
