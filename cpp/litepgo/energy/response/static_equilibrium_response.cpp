#include "litepgo/energy/response/static_equilibrium_response.hpp"
#include "litepgo/linalg/sparse_solver/auto_sparse_solver.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo::energy {
namespace {
SpMat state_pattern(int parameters, const dof::DofReduction& fixed) {
    if (parameters <= 0)
        throw std::invalid_argument("StaticEquilibriumResponse: positive parameter count required");
    // shortcut: equilibrium sensitivities are generally dense; use adjoints for large parameter fields.
    SpMat pattern(fixed.num_dofs(), parameters);
    for (int c = 0; c < parameters; ++c)
        for (int r : fixed.free_dofs())
            pattern.insert(r, c) = 1;
    pattern.makeCompressed();
    return pattern;
}
} // namespace
StaticEquilibriumResponse::StaticEquilibriumResponse(int num_parameters, dof::DofReduction fixed, Vec initial,
                                                     solver::NewtonOptions options, MakeEnergy make_energy,
                                                     Partial partial)
    : FixedPatternResponse(state_pattern(num_parameters, fixed)), fixed_(std::move(fixed)),
      initial_(std::move(initial)), options_(options), make_energy_(std::move(make_energy)),
      partial_(std::move(partial)) {
    if (initial_.size() != fixed_.num_dofs() || !initial_.allFinite() || !make_energy_ || !partial_ ||
        !std::isfinite(options_.gradient_tolerance) || options_.gradient_tolerance < 0 || options_.max_iterations < 0 ||
        options_.max_line_search_steps < 1 || !(options_.line_search_shrink > 0 && options_.line_search_shrink < 1))
        throw std::invalid_argument("StaticEquilibriumResponse: invalid initial state, options or callbacks");
}
Vec StaticEquilibriumResponse::eval_values(const ConstVecRef& p, SpMat* J) const {
    const Vec parameters = p;
    auto potential = make_energy_(parameters);
    if (!potential || potential->num_dofs() != num_outputs())
        throw std::invalid_argument("StaticEquilibriumResponse: incompatible energy");
    linalg::AutoIndefiniteSparseSolver linear_solver;
    const auto result = solver::newton_solve(*potential, fixed_, initial_, options_, linear_solver);
    if (result.status != solver::NewtonStatus::Converged)
        throw std::domain_error("StaticEquilibriumResponse: equilibrium did not converge");
    if (J && fixed_.num_free() > 0) {
        SpMat B = partial_(parameters, result.x);
        B.makeCompressed();
        if (B.rows() != num_outputs() || B.cols() != num_inputs() ||
            !Eigen::Map<const Vec>(B.valuePtr(), B.nonZeros()).allFinite())
            throw std::invalid_argument("StaticEquilibriumResponse: invalid residual parameter Jacobian");
        potential->update_pattern(result.x);
        const SpMat K = fixed_.reduce(potential->eval_hessian(result.x, HessianKind::Exact));
        if (!linear_solver.compute(K))
            throw std::domain_error("StaticEquilibriumResponse: singular equilibrium stiffness");
        for (int c = 0; c < num_inputs(); ++c) {
            const Vec rhs = -fixed_.reduce(Vec(B.col(c)));
            const Vec sensitivity = linear_solver.solve(rhs);
            if (!sensitivity.allFinite() || (K * sensitivity - rhs).norm() > 1e-8 * (1 + rhs.norm()))
                throw std::domain_error("StaticEquilibriumResponse: inaccurate sensitivity solve");
            for (SpMat::InnerIterator it(*J, c); it; ++it)
                it.valueRef() = sensitivity[fixed_.free_index(static_cast<int>(it.row()))];
        }
    }
    return result.x;
}
} // namespace litepgo::energy
