#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>
namespace litepgo::energy {
Energy::Energy(int n) : num_dofs_(n) {
    if (n < 0)
        throw std::invalid_argument("Energy requires a non-negative DOF count");
}
EnergyEvaluation::EnergyEvaluation(const Energy& energy, Vec x) : energy_(energy), x_(std::move(x)) {}
int EnergyEvaluation::num_dofs() const noexcept {
    return energy_.num_dofs();
}
std::shared_ptr<EnergyEvaluation> Energy::evaluate(const ConstVecRef& x) const {
    if (x.size() != num_dofs_ || !x.allFinite())
        throw std::invalid_argument("Energy::evaluate: invalid input");
    auto result = make_evaluation(x);
    if (!result || result->num_dofs() != num_dofs_)
        throw std::logic_error("Energy: invalid evaluation");
    return result;
}
std::shared_ptr<EnergyEvaluation> Energy::make_evaluation(const ConstVecRef& x) const {
    return std::shared_ptr<EnergyEvaluation>(new EnergyEvaluation(*this, x));
}
EnergyValues Energy::eval_impl(const ConstVecRef&, bool, bool) const {
    throw std::logic_error("Energy: missing evaluation implementation");
}
EnergyValues EnergyEvaluation::compute(bool g, bool h) {
    return energy_.eval_impl(x_, g, h);
}
EnergyValues EnergyEvaluation::compute_projected() {
    return energy_.eval_projected_values(x_, *this);
}
void EnergyEvaluation::save(EnergyValues result, bool gradient, bool hessian, bool projected) {
    const int num_dofs_ = num_dofs();
    if (!std::isfinite(result.value))
        throw std::logic_error("Energy: values must be finite");
    if (result.gradient.has_value() != gradient || result.hessian.has_value() != hessian)
        throw std::logic_error("Energy: requested outputs missing or unexpected outputs returned");
    if (result.gradient && (result.gradient->size() != num_dofs_ || !result.gradient->allFinite()))
        throw std::logic_error("Energy: invalid gradient");
    if (result.hessian) {
        const auto& H = *result.hessian;
#ifndef NDEBUG
        if (!linalg::is_canonical(H))
            throw std::logic_error("Energy: invalid Hessian structure");
#endif
        if (H.rows() != num_dofs_ || H.cols() != num_dofs_ || !H.coeffs().allFinite())
            throw std::logic_error("Energy: invalid Hessian");
#ifndef NDEBUG
        if (!linalg::same_pattern(H, SpMat(H.transpose())) || (H.rows() && !H.isApprox(SpMat(H.transpose()), 1e-12)))
            throw std::logic_error("Energy: invalid Hessian structure or symmetry");
#endif
    }

    value_ = result.value;
    if (result.gradient)
        gradient_ = std::move(result.gradient);
    if (result.hessian)
        (projected ? projected_ : hessian_) = std::move(result.hessian);
}
double EnergyEvaluation::value() {
    if (!value_)
        save(compute(false, false), false, false, false);
    return *value_;
}
Vec EnergyEvaluation::gradient() {
    if (!gradient_)
        save(compute(true, false), true, false, false);
    return *gradient_;
}
SpMat EnergyEvaluation::hessian() {
    if (!hessian_)
        save(compute(true, true), true, true, false);
    return *hessian_;
}
SpMat EnergyEvaluation::projected_hessian() {
    if (!projected_)
        save(compute_projected(), true, true, true);
    return *projected_;
}
EnergyValues Energy::eval_projected_values(const ConstVecRef&, EnergyEvaluation& exact) const {
    SpMat projected = project_hessian(exact.hessian());
    return {exact.value(), exact.gradient(), std::move(projected)};
}
SpMat project_hessian(const SpMat& H) {
    if (!H.rows())
        return H;
    const Eigen::MatrixXd dense(H);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver((0.5 * dense + 0.5 * dense.transpose()).eval());
    if (solver.info() != Eigen::Success)
        throw std::domain_error("Energy: Hessian eigendecomposition failed");
    if (solver.eigenvalues().minCoeff() >= 0.)
        return H;
    const Eigen::MatrixXd projected =
        solver.eigenvectors() * solver.eigenvalues().cwiseMax(0.).asDiagonal() * solver.eigenvectors().transpose();
    if (!projected.allFinite())
        throw std::domain_error("Energy: nonfinite projected Hessian");
    const Eigen::MatrixXd symmetric = 0.5 * projected + 0.5 * projected.transpose();
    return symmetric.sparseView();
}
double EnergyEvaluation::feasible_step_impl(const ConstVecRef& dx) const {
    return energy_.feasible_step(x_, dx);
}
double EnergyEvaluation::max_feasible_step(const ConstVecRef& dx) const {
    if (dx.size() != num_dofs() || !dx.allFinite())
        throw std::invalid_argument("EnergyEvaluation: invalid direction");
    const double alpha = feasible_step_impl(dx);
    if (!(alpha >= 0. && alpha <= 1.))
        throw std::logic_error("Energy: invalid feasible step");
    return alpha;
}
double Energy::feasible_step(const ConstVecRef&, const ConstVecRef&) const {
    return 1.;
}
} // namespace litepgo::energy
