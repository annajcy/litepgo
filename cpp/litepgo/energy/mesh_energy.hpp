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
// x_local = map.gather(e, x) preserves the map's matrix or vector shape.
// Derivatives use the corresponding flattened scalar-slot order, without repacking.
// Numerical overflow in projection or assembly throws std::domain_error; invalid
// callback outputs (including non-finite values) remain logic_error.
class MeshEnergy : public FixedPatternEnergy {
  public:
    // Only for energies feasible along every finite segment: step defaults to 1.
    // eval_local(e, x_local, gradient*, hessian*) returns integrated energy.
    // Non-null outputs must be overwritten, in the map's local DOF order.
    // Unsupported derivative requests must throw; pointers are independently nullable.
    template <class DofMap, class Lambda>
    MeshEnergy(const DofMap& map, Lambda eval_local)
        : MeshEnergy(map, std::move(eval_local), [](int, const auto&, const auto&) { return 1.0; }) {}

    // step(e, x_local, dx_local) returns a safe inclusive bound in [0,1].
    // Trial values are x_local + alpha * dx_local (delta is unscaled).
    // It must reject infeasible starting states, e.g. by throwing std::domain_error.
    // Like eval_local, step may run concurrently on different elements.
    template <class DofMap, class Lambda, class Step>
    MeshEnergy(const DofMap& map, Lambda eval_local, Step step) : MeshEnergy(make_assembler(map)) {
        auto owned_map = std::make_shared<const DofMap>(map);
        eval_local_ = [owned_map, eval_local = std::move(eval_local)](int element, const ConstVecRef& x, Vec* gradient,
                                                                      Eigen::MatrixXd* hessian) {
            const auto x_local = owned_map->gather(element, x);
            return eval_local(element, x_local, gradient, hessian);
        };
        local_step_ = [owned_map, step = std::move(step)](int element, const ConstVecRef& x, const ConstVecRef& dx) {
            const auto x_local = owned_map->gather(element, x);
            const auto dx_local = owned_map->gather(element, dx);
            return step(element, x_local, dx_local);
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
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const override;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;

    Assembler assembler_;
    std::function<double(int, const ConstVecRef&, Vec*, Eigen::MatrixXd*)> eval_local_;
    std::function<double(int, const ConstVecRef&, const ConstVecRef&)> local_step_;
    mutable std::vector<double> local_values_;
    mutable std::vector<Vec> local_gradients_;
    mutable std::vector<Eigen::MatrixXd> local_hessians_;
};
} // namespace litepgo
