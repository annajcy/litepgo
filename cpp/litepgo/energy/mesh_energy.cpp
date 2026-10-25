#include "litepgo/energy/mesh_energy.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

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
    const auto count = static_cast<std::size_t>(assembler_.num_elements());
    local_values_.resize(count);
    if (gradient)
        local_gradients_.resize(count);
    if (hessian)
        local_hessians_.resize(count);
    try {
        tbb::parallel_for(0, assembler_.num_elements(), [&](int e) {
            const auto index = static_cast<std::size_t>(e);
            auto* g = gradient ? &local_gradients_[index] : nullptr;
            auto* h = hessian ? &local_hessians_[index] : nullptr;
            const auto n = static_cast<Eigen::Index>(assembler_.dof_indices(e).size());
            // Reused buffers must still detect incomplete callback writes.
            const double nan = std::numeric_limits<double>::quiet_NaN();
            if (g)
                g->setConstant(n, nan);
            if (h)
                h->setConstant(n, n, nan);
            const double value = evaluate_local_(e, x, g, h);
            if (!std::isfinite(value))
                throw std::logic_error("MeshEnergy: non-finite local energy");
            if (g && (g->size() != n || !g->allFinite()))
                throw std::logic_error("MeshEnergy: invalid local gradient");
            if (h) {
                if (h->rows() != n || h->cols() != n || !h->allFinite())
                    throw std::logic_error("MeshEnergy: invalid local Hessian");
                if (kind == HessianKind::Projected) {
                    project_hessian(*h);
                    if (!h->allFinite())
                        throw std::logic_error("MeshEnergy: non-finite projected Hessian");
                }
            }
            local_values_[index] = value;
        });
    } catch (...) {
        local_gradients_.clear();
        local_hessians_.clear();
        throw;
    }
    assembler_.assemble(local_gradients_, local_hessians_, gradient, hessian);
    // Retain the original element order for the scalar energy sum.
    const double value = std::accumulate(local_values_.begin(), local_values_.end(), 0.0);
    return value;
}

double MeshEnergy::feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    if (!x.allFinite() || !dx.allFinite())
        throw std::invalid_argument("MeshEnergy requires finite DOFs and DOF deltas");
    return tbb::parallel_reduce(
        tbb::blocked_range<int>(0, assembler_.num_elements(), 16), 1.0,
        [&](const auto& range, double result) {
            for (int e = range.begin(); e < range.end(); ++e) {
                const double step = local_step_(e, x, dx);
                if (!(step >= 0 && step <= 1))
                    throw std::logic_error("MeshEnergy: local feasible step outside [0,1]");
                result = std::min(result, step);
            }
            return result;
        },
        [](double a, double b) { return std::min(a, b); });
}
} // namespace litepgo
