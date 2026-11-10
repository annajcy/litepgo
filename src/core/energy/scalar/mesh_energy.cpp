#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/common/busy_scope.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace litepgo::energy {
namespace {
void project_local_hessian(Eigen::Ref<BatchMatrix> hessian) {
    if (hessian.rows() == 0)
        return;
    const Eigen::MatrixXd symmetric = 0.5 * hessian + 0.5 * hessian.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetric);
    if (solver.info() != Eigen::Success)
        throw std::domain_error("MeshEnergy: local Hessian eigendecomposition failed");
    hessian =
        solver.eigenvectors() * solver.eigenvalues().cwiseMax(0.0).asDiagonal() * solver.eigenvectors().transpose();
    if (!hessian.allFinite())
        throw std::domain_error("MeshEnergy: non-finite projected Hessian");
}
} // namespace

MeshEnergy::MeshEnergy(dof::EnergyAssembler assembler, int batch_size)
    : FixedPatternEnergy(assembler.hessian_pattern()), assembler_(std::move(assembler)) {
    if (batch_size <= 0)
        throw std::invalid_argument("MeshEnergy: invalid batch size");
    std::map<int, std::vector<int>> groups;
    for (int e = 0; e < assembler_.num_elements(); ++e)
        groups[static_cast<int>(assembler_.dof_indices(e).size())].push_back(e);
    for (const auto& [size, ids] : groups)
        for (std::size_t begin = 0; begin < ids.size(); begin += batch_size)
            batches_.push_back({.ids = {ids.begin() + begin, ids.begin() + std::min(ids.size(), begin + batch_size)}});
    values_.resize(assembler_.num_elements());
    views_.resize(assembler_.num_elements());
}
void MeshEnergy::gather(const std::vector<int>& ids, const ConstVecRef& x, BatchMatrix& result) const {
    result.resize(ids.size(), assembler_.dof_indices(ids.front()).size());
    for (std::size_t b = 0; b < ids.size(); ++b)
        result.row(b) = x(assembler_.dof_indices(ids[b])).transpose();
}
double MeshEnergy::eval_values(const ConstVecRef& x, Vec* g, SpMat* H) const {
    return assemble_values(x, g, H, false);
}
energy::EnergyValues MeshEnergy::eval_projected_values(const ConstVecRef& x, energy::EnergyEvaluation&) const {
    energy::EnergyValues result{0., Vec(num_dofs()), make_hessian_pattern(x)};
    result.value = assemble_values(x, &*result.gradient, &*result.hessian, true);
    return result;
}
double MeshEnergy::assemble_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, bool project) const {
    ::litepgo::detail::BusyScope scope(busy_, "MeshEnergy");
    tbb::parallel_for(std::size_t(0), batches_.size(), [&](std::size_t k) {
        auto& data = batches_[k];
        const auto& ids = data.ids;
        gather(ids, x, data.inputs);
        const auto& inputs = data.inputs;
        const Eigen::Index count = inputs.rows(), n = inputs.cols();
        auto& g = data.gradient;
        auto& H = data.hessian;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (gradient)
            g.setConstant(count, n, nan);
        if (hessian)
            H.setConstant(count * n, n, nan);
        data.values = eval_batch(ids, inputs, gradient ? &g : nullptr, hessian ? &H : nullptr);
        const auto& values = data.values;
        if (values.size() != count || !values.allFinite() ||
            (gradient && (g.rows() != count || g.cols() != n || !g.allFinite())) ||
            (hessian && (H.rows() != count * n || H.cols() != n || !H.allFinite())))
            throw std::logic_error("MeshEnergy: invalid batch outputs");
        for (Eigen::Index b = 0; b < count; ++b) {
            const int e = ids[b];
            values_[e] = values[b];
            if (hessian && project)
                project_local_hessian(H.middleRows(b * n, n));
            if (gradient || hessian) {
                const Eigen::Map<const Vec> local_g(gradient && n ? g.data() + b * n : nullptr, gradient ? n : 0);
                const Eigen::Map<const BatchMatrix> local_H(hessian && n ? H.data() + b * n * n : nullptr,
                                                            hessian ? n : 0, hessian ? n : 0);
                views_[e] = dof::EnergyAssembler::ElementView(local_g, local_H);
            }
        }
    });
    assembler_.assemble(views_, gradient, hessian);
    const double value = std::accumulate(values_.begin(), values_.end(), 0.);
    if (!std::isfinite(value))
        throw std::domain_error("MeshEnergy: non-finite energy sum");
    return value;
}
double MeshEnergy::feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    ::litepgo::detail::BusyScope scope(busy_, "MeshEnergy");
    return tbb::parallel_reduce(
        tbb::blocked_range<std::size_t>(0, batches_.size()), 1.,
        [&](const auto& range, double bound) {
            for (auto k = range.begin(); k < range.end(); ++k) {
                auto& data = batches_[k];
                const auto& ids = data.ids;
                gather(ids, x, data.inputs);
                gather(ids, dx, data.delta);
                const Vec steps = eval_feasible_step_batch(ids, data.inputs, data.delta);
                if (steps.size() != static_cast<Eigen::Index>(ids.size()) || !steps.allFinite() ||
                    (steps.array() < 0).any() || (steps.array() > 1).any())
                    throw std::logic_error("MeshEnergy: invalid batch steps");
                bound = std::min(bound, steps.minCoeff());
            }
            return bound;
        },
        [](double a, double b) { return std::min(a, b); });
}
} // namespace litepgo::energy
