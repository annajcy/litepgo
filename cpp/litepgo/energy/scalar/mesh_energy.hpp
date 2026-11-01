#pragma once
#include "litepgo/dof/energy_assembler.hpp"
#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"
#include <functional>
#include <stdexcept>
#include <utility>

namespace litepgo::energy {
using BatchMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
// Concurrent callbacks on size-grouped batches; input rows follow map DOF-index
// order. Values have B entries, gradients B x n, Hessians (B*n) x n. Callbacks
// must overwrite requested outputs and safely share captures. No ordering assumed.
// batch_size=1 uses exactly the same protocol. Projection is per element, followed
// by deterministic EnergyAssembler reduction. Maps are snapshotted at construction.
class MeshEnergy : public FixedPatternEnergy {
  public:
    using Callback = std::function<Vec(std::span<const int>, const BatchMatrix&, BatchMatrix*, BatchMatrix*)>;
    using Step = std::function<Vec(std::span<const int>, const BatchMatrix&, const BatchMatrix&)>;
    // Omit step only when every finite segment is feasible. Otherwise return one
    // safe bound in [0,1] per batch element. Step callbacks may also run concurrently.
    template <class Map>
    MeshEnergy(const Map& map, Callback callback, int batch_size = 256, Step step = {})
        : MeshEnergy(make_assembler(map), std::move(callback), batch_size, std::move(step)) {}

  private:
    template <class Map> static dof::EnergyAssembler make_assembler(const Map& map) {
        if (map.num_elements() < 0)
            throw std::invalid_argument("MeshEnergy: negative element count");
        std::vector<std::vector<int>> indices;
        for (int e = 0; e < map.num_elements(); ++e)
            indices.push_back(map.dof_indices(e));
        return dof::EnergyAssembler(map.num_dofs(), std::move(indices));
    }
    MeshEnergy(dof::EnergyAssembler assembler, Callback callback, int batch_size, Step step);
    void gather(const std::vector<int>& ids, const ConstVecRef& x, BatchMatrix& result) const;
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const override;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;
    dof::EnergyAssembler assembler_;
    Callback callback_;
    Step step_;
    std::vector<std::vector<int>> batches_;
    struct Storage {
        BatchMatrix inputs, gradient, hessian, delta;
        Vec values;
    };
    mutable std::vector<Storage> storage_;
    mutable std::vector<double> values_;
    mutable std::vector<dof::EnergyAssembler::ElementView> views_;
};
} // namespace litepgo::energy
