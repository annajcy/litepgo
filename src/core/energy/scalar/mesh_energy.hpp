#pragma once
#include "litepgo/dof/energy_assembler.hpp"
#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"
#include <atomic>
#include <stdexcept>
#include <utility>

namespace litepgo::energy {
using BatchMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// Concurrent batch hooks on size-grouped batches; input rows follow map DOF-index
// order. Values have B entries, gradients B x n, Hessians (B*n) x n. Implementations
// must overwrite requested outputs and safely share state. No ordering assumed.
// batch_size=1 uses exactly the same protocol. Projection is per element, followed
// by deterministic EnergyAssembler reduction. Maps are snapshotted at construction.
class MeshEnergy : public FixedPatternEnergy {
  public:
    template <class Map>
    MeshEnergy(const Map& map, int batch_size = 256) : MeshEnergy(make_assembler(map), batch_size) {}

  protected:
    virtual Vec eval_batch(std::span<const int> ids, const BatchMatrix& x, BatchMatrix* g, BatchMatrix* H) const = 0;
    // Override for restricted domains: return a safe [0,1] bound per element.
    virtual Vec eval_feasible_step_batch(std::span<const int> ids, const BatchMatrix&, const BatchMatrix&) const {
        return Vec::Ones(ids.size());
    }

  private:
    template <class Map> static dof::EnergyAssembler make_assembler(const Map& map) {
        if (map.num_elements() < 0)
            throw std::invalid_argument("MeshEnergy: negative element count");
        std::vector<std::vector<int>> indices;
        for (int e = 0; e < map.num_elements(); ++e)
            indices.push_back(map.dof_indices(e));
        return dof::EnergyAssembler(map.num_dofs(), std::move(indices));
    }
    MeshEnergy(dof::EnergyAssembler assembler, int batch_size);
    void gather(const std::vector<int>& ids, const ConstVecRef& x, BatchMatrix& result) const;
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian) const override;
    energy::EnergyValues eval_projected_values(const ConstVecRef& x, energy::EnergyEvaluation&) const override;
    double assemble_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, bool project) const;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;
    dof::EnergyAssembler assembler_;
    struct Batch {
        std::vector<int> ids;
        BatchMatrix inputs, gradient, hessian, delta;
        Vec values;
    };
    mutable std::atomic<bool> busy_ = false;
    mutable std::vector<Batch> batches_;
    mutable std::vector<double> values_;
    mutable std::vector<dof::EnergyAssembler::ElementView> views_;
};
} // namespace litepgo::energy
