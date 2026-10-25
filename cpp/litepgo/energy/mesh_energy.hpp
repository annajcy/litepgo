#pragma once

#include "litepgo/dof/assembler.hpp"
#include "litepgo/energy/fixed_pattern_energy.hpp"
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace litepgo {

// Owns a fixed DOF plan and callbacks, not references to the supplied map. Every call
// uses x's current values. Callbacks must support concurrent calls on different
// elements and own (or safely share) their captures. Callbacks must be copyable.
// Built-in maps use vertex-major matrices or column vectors; local_dof_values preserves
// those shapes. Derivatives use the corresponding flattened scalar-slot order.
class MeshEnergy final : public FixedPatternEnergy {
  public:
    // Only for energies feasible along every finite segment: step defaults to 1.
    // evaluate_local(e, local_dof_values, gradient*, hessian*) returns integrated energy.
    // Non-null outputs must be overwritten, in the map's local DOF order.
    // Unsupported derivative requests must throw; pointers are independently nullable.
    template <class DofMap, class Lambda>
    MeshEnergy(const DofMap& map, Lambda evaluate_local)
        : MeshEnergy(map, std::move(evaluate_local), [](int, const auto&, const auto&) { return 1.0; }) {}

    // step(e, local_dof_values, local_dof_delta) returns a safe inclusive bound in [0,1].
    // Trial values are local_dof_values + alpha * local_dof_delta (delta is unscaled).
    // It must reject infeasible starting states, e.g. by throwing InfeasibleState.
    template <class DofMap, class Lambda, class Step>
    MeshEnergy(const DofMap& map, Lambda evaluate_local, Step step) : MeshEnergy(make_assembler(map)) {
        auto owned_map = std::make_shared<const DofMap>(map);
        evaluate_local_ = [owned_map, evaluate_local = std::move(evaluate_local)](
                              int element, const ConstVecRef& x, Vec* gradient, Eigen::MatrixXd* hessian) {
            const auto local_dof_values = owned_map->gather(element, x);
            return evaluate_local(element, local_dof_values, gradient, hessian);
        };
        local_step_ = [owned_map, step = std::move(step)](int element, const ConstVecRef& x, const ConstVecRef& dx) {
            const auto local_dof_values = owned_map->gather(element, x);
            const auto local_dof_delta = owned_map->gather(element, dx);
            return step(element, local_dof_values, local_dof_delta);
        };
    }

  private:
    template <class DofMap> static Assembler make_assembler(const DofMap& map) {
        std::vector<std::vector<int>> indices;
        for (int element = 0; element < map.num_elements(); ++element)
            indices.push_back(map.dof_indices(element));
        return Assembler(map.num_dofs(), std::move(indices));
    }

    explicit MeshEnergy(Assembler assembler);
    double evaluate_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const override;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;

    Assembler assembler_;
    std::function<double(int, const ConstVecRef&, Vec*, Eigen::MatrixXd*)> evaluate_local_;
    std::function<double(int, const ConstVecRef&, const ConstVecRef&)> local_step_;
};
} // namespace litepgo
