#pragma once

#include "litepgo/energy/energy.h"

#include <memory>
#include <vector>

namespace litepgo {

struct WeightedEnergy {
    std::shared_ptr<Energy> energy;
    double weight = 1.0;
};

// E(x) = sum_i w_i E_i(x), over terms that share the same DOFs (wrap a term in an
// AffineMappedEnergy to act on a subset). Weights are finite and nonzero; they may
// be negative. The pattern is fixed if every term's is.
//
// The projected Hessian is sum_i w_i P_i, so it is PSD only when every weight is
// positive and every term's P_i is. A term's InfeasibleState passes through the
// sum whatever its weight. The feasible step is the smallest over terms.
//
// The Hessian pattern is the union of the terms' patterns. Each term has its own
// Hessian buffer, accumulated into the union in a fixed order, so results are
// deterministic.
class WeightedEnergySum final : public Energy {
  public:
    // At least one term, all with the same num_dofs().
    explicit WeightedEnergySum(std::vector<WeightedEnergy> terms);

  private:
    struct Setup {
        std::vector<WeightedEnergy> terms;
        SpMat pattern;
        bool fixed_pattern;
    };
    static Setup checked_setup(std::vector<WeightedEnergy> terms);
    explicit WeightedEnergySum(Setup setup);

    std::optional<SpMat> refresh_pattern(const ConstVecRef& x) override;
    double evaluate_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const override;
    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override;

    void rebuild_positions(const SpMat& pattern);

    struct TermScratch {
        // The term's Hessian buffer. Its pattern is the one positions was built
        // from, so comparing it with the term's current pattern detects staleness.
        SpMat hessian;
        Vec gradient;
        // positions[k]: where the term's k-th stored entry sits in the union.
        std::vector<int> positions;
    };

    std::vector<WeightedEnergy> terms_;
    mutable std::vector<TermScratch> scratch_;
};

} // namespace litepgo
