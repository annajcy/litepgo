#pragma once

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include <map>
#include <span>

namespace litepgo::energy {
using BatchMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Fixed vertex stencils, grouped by size. Batch callbacks run serially;
// MeshEnergy retains the per-term validation, projection and assembly contract.
class BatchedMeshEnergy final : public FixedPatternEnergy {
  public:
    using Callback = std::function<Vec(std::span<const int>, const BatchMatrix&, BatchMatrix*, BatchMatrix*)>;
    using Step = std::function<Vec(std::span<const int>, const BatchMatrix&, const BatchMatrix&)>;
    BatchedMeshEnergy(const dof::StencilDofMap& map, Callback callback, int batch_size = 256, Step step = {})
        : FixedPatternEnergy(map.num_dofs(), pattern(map)), map_(map), callback_(std::move(callback)),
          step_(std::move(step)) {
        if (batch_size <= 0 || !callback_)
            throw std::invalid_argument("Invalid batch size or callback");
        std::map<int, std::vector<int>> groups;
        for (int e = 0; e < map.num_elements(); ++e)
            groups[static_cast<int>(map.vertex_indices(e).size())].push_back(e);
        for (const auto& [size, ids] : groups)
            for (std::size_t begin = 0; begin < ids.size(); begin += batch_size)
                batches_.emplace_back(ids.begin() + begin, ids.begin() + std::min(ids.size(), begin + batch_size));
        values_.resize(map.num_elements());
        gradients_.resize(map.num_elements());
        hessians_.resize(map.num_elements());
        // shortcut: reuse assembly with an extra gather/copy pass; split out shared assembly if profiling warrants it.
        assembled_ = std::make_unique<MeshEnergy>(map, [this](int e, const auto&, Vec* g, Eigen::MatrixXd* h) {
            if (g)
                *g = gradients_[e];
            if (h)
                *h = hessians_[e];
            return values_[e];
        });
    }

  private:
    static SpMat pattern(const dof::StencilDofMap& map) {
        std::vector<std::vector<int>> indices;
        for (int e = 0; e < map.num_elements(); ++e)
            indices.push_back(map.dof_indices(e));
        return dof::Assembler(map.num_dofs(), std::move(indices)).hessian_pattern();
    }
    BatchMatrix gather(const std::vector<int>& ids, const ConstVecRef& x) const {
        const int m = static_cast<int>(map_.dof_indices(ids.front()).size());
        BatchMatrix result(ids.size(), m);
        for (std::size_t b = 0; b < ids.size(); ++b)
            result.row(b) = map_.gather(ids[b], x).reshaped<Eigen::RowMajor>().transpose();
        return result;
    }
    double eval_values(const ConstVecRef& x, Vec* g, SpMat* h, HessianKind kind) const override {
        if (!x.allFinite())
            throw std::invalid_argument("BatchedMeshEnergy requires finite DOFs");
        for (const auto& ids : batches_) {
            const auto local = gather(ids, x);
            const Eigen::Index count = local.rows(), m = local.cols();
            BatchMatrix gradients, hessians;
            const Vec values = callback_(ids, local, g ? &gradients : nullptr, h ? &hessians : nullptr);
            if (values.size() != count || !values.allFinite() ||
                (g && (gradients.rows() != count || gradients.cols() != m || !gradients.allFinite())) ||
                (h && (hessians.rows() != count * m || hessians.cols() != m || !hessians.allFinite())))
                throw std::logic_error("BatchedMeshEnergy: invalid batch outputs");
            for (Eigen::Index b = 0; b < count; ++b) {
                values_[ids[b]] = values[b];
                if (g)
                    gradients_[ids[b]] = gradients.row(b).transpose();
                if (h)
                    hessians_[ids[b]] = hessians.middleRows(b * m, m);
            }
        }
        return assembled_->eval(x, g, h, kind);
    }
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override {
        if (!x.allFinite() || !dx.allFinite())
            throw std::invalid_argument("BatchedMeshEnergy requires finite DOFs");
        double result = 1.;
        if (step_)
            for (const auto& ids : batches_) {
                const Vec steps = step_(ids, gather(ids, x), gather(ids, dx));
                if (steps.size() != static_cast<Eigen::Index>(ids.size()) || !steps.allFinite() ||
                    (steps.array() < 0).any() || (steps.array() > 1).any())
                    throw std::logic_error("BatchedMeshEnergy: invalid batch steps");
                result = std::min(result, steps.minCoeff());
            }
        return result;
    }
    dof::StencilDofMap map_;
    Callback callback_;
    Step step_;
    std::vector<std::vector<int>> batches_;
    mutable Vec values_;
    mutable std::vector<Vec> gradients_;
    mutable std::vector<Eigen::MatrixXd> hessians_;
    std::unique_ptr<MeshEnergy> assembled_;
};
} // namespace litepgo::energy
