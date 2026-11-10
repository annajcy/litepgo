#pragma once

#include "litepgo/energy/scalar/energy.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace litepgo::energy {

// E(z) = F(T z + offset): the inner energy F seen through an affine map, such as
// periodic or kinematic boundary conditions acting on independent fluctuations.
// The gradient is T^T g and the Hessian T^T H T.
//
// T^T H T is not formed by sparse products. A contribution table, built once per
// inner pattern, lists for each stored entry of the result which entries of H
// contribute and with which coefficient T(i, a) T(j, b); evaluating sums over it
// using independent output storage. With T's rows holding one or two entries (selection and
// periodic maps), the table is about as long as H has stored entries.
class AffineMappedEnergy final : public Energy {
  public:
    // T must have energy->num_dofs() rows; z has T.cols() entries. An empty offset
    // means zero. Independence of T's columns is not checked.
    AffineMappedEnergy(std::shared_ptr<Energy> energy, SpMat T, Vec offset = Vec());

  private:
    struct Contribution {
        int output;
        int input;
        double coefficient;
    };
    struct HessianPlan {
        SpMat pattern;
        std::vector<Contribution> contributions;
    };
    static HessianPlan plan_hessian(const SpMat& inner_pattern, const SpMat& map);

    class Evaluation;
    std::shared_ptr<EnergyEvaluation> make_evaluation(const ConstVecRef& x) const override;
    std::shared_ptr<Energy> energy_;
    SpMat map_;
    Vec offset_;
    struct CachedPlan {
        SpMat inner_pattern;
        HessianPlan hessian;
    };
    mutable std::optional<CachedPlan> plan_;
    mutable std::atomic<bool> busy_ = false;
};
} // namespace litepgo::energy
