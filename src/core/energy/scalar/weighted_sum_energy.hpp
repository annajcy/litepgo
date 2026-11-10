#pragma once

#include "litepgo/energy/scalar/energy.hpp"

#include <atomic>
#include <memory>
#include <vector>

namespace litepgo::energy {

struct WeightedEnergy {
    std::shared_ptr<Energy> energy;
    double weight = 1.0;
};

// E(x) = sum_i w_i E_i(x), over terms that share the same DOFs (wrap a term in an
// AffineMappedEnergy to act on a subset). Weights are finite and nonzero; they may
// be negative.
//
// Positive weights combine child PSD approximations; negative weights use
// spectral projection of the exact summed Hessian. A term's std::domain_error passes through the
// sum whatever its weight. The feasible step is the smallest over terms.
//
// The Hessian pattern is the union of the terms' patterns. Each term has its own
// independently returned Hessian, accumulated in a fixed order, so results are
// deterministic.
class WeightedSumEnergy final : public Energy {
  public:
    // At least one term, all with the same num_dofs().
    explicit WeightedSumEnergy(std::vector<WeightedEnergy> terms);

  private:
    static int checked_size(const std::vector<WeightedEnergy>& terms);
    class Evaluation;
    std::shared_ptr<EnergyEvaluation> make_evaluation(const ConstVecRef& x, bool borrow_input) const override;
    struct TermPlan {
        SpMat pattern;
        std::vector<int> positions;
    };
    struct AssemblyPlan {
        SpMat pattern;
        std::vector<TermPlan> terms;
    };
    std::vector<WeightedEnergy> terms_;
    mutable std::optional<AssemblyPlan> plan_;
    mutable std::atomic<bool> busy_ = false;
};
} // namespace litepgo::energy
