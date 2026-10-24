#include "litepgo/fem/assembly/mesh_energy.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tbb/parallel_for.h>

namespace litepgo {
namespace {
void project_hessian(Eigen::MatrixXd& hessian) {
    if (hessian.rows() == 0)
        return;
    const Eigen::MatrixXd symmetric = 0.5 * hessian + 0.5 * hessian.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
    if (solver.info() != Eigen::Success)
        throw std::logic_error("MeshEnergy: local Hessian eigendecomposition failed");
    hessian =
        solver.eigenvectors() * solver.eigenvalues().cwiseMax(0.0).asDiagonal() * solver.eigenvectors().transpose();
}
} // namespace

MeshEnergy::MeshEnergy(Assembler assembler)
    : FixedPatternEnergy(static_cast<int>(assembler.hessian_pattern().rows()), assembler.hessian_pattern()),
      assembler_(std::move(assembler)) {}

double MeshEnergy::evaluate_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const {
    if (!x.allFinite())
        throw std::invalid_argument("MeshEnergy requires finite DOFs");
    struct Result {
        double value;
        Vec gradient;
        Eigen::MatrixXd hessian;
    };
    std::vector<Result> results(static_cast<std::size_t>(assembler_.num_elements()));
    tbb::parallel_for(0, assembler_.num_elements(), [&](int e) {
        auto& result = results[static_cast<std::size_t>(e)];
        const auto n = static_cast<Eigen::Index>(assembler_.dof_indices(e).size());
        // Unwritten output entries must fail validation rather than read uninitialized memory.
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (gradient)
            result.gradient = Vec::Constant(n, nan);
        if (hessian)
            result.hessian = Eigen::MatrixXd::Constant(n, n, nan);
        result.value =
            evaluate_local_(e, x, gradient ? &result.gradient : nullptr, hessian ? &result.hessian : nullptr);
        if (!std::isfinite(result.value))
            throw std::logic_error("MeshEnergy: non-finite local energy");
        if (gradient && (result.gradient.size() != n || !result.gradient.allFinite()))
            throw std::logic_error("MeshEnergy: invalid local gradient");
        if (hessian) {
            if (result.hessian.rows() != n || result.hessian.cols() != n || !result.hessian.allFinite())
                throw std::logic_error("MeshEnergy: invalid local Hessian");
            if (kind == HessianKind::Projected)
                project_hessian(result.hessian);
        }
    });
    if (gradient)
        gradient->setZero();
    if (hessian)
        std::fill_n(hessian->valuePtr(), hessian->nonZeros(), 0.0);
    double value = 0;
    for (int e = 0; e < assembler_.num_elements(); ++e) {
        const auto& result = results[static_cast<std::size_t>(e)];
        value += result.value;
        if (gradient)
            assembler_.add_gradient(e, result.gradient, *gradient);
        if (hessian)
            assembler_.add_hessian(e, result.hessian, *hessian);
    }
    return value;
}

double MeshEnergy::feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    if (!x.allFinite() || !dx.allFinite())
        throw std::invalid_argument("MeshEnergy requires finite DOFs and DOF deltas");
    double result = 1;
    for (int e = 0; e < assembler_.num_elements(); ++e) {
        const double step = local_step_(e, x, dx);
        if (!(step >= 0 && step <= 1))
            throw std::logic_error("MeshEnergy: local feasible step outside [0,1]");
        result = std::min(result, step);
    }
    return result;
}
} // namespace litepgo
