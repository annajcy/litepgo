#include "litepgo/energy/scalar/weighted_sum_energy.hpp"

#include "litepgo/common/busy_scope.hpp"
#include "litepgo/energy/scalar/detail/evaluation_access.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace litepgo::energy {

namespace {

// The union of n x n patterns: every position stored by any of them, once.
SpMat merged_pattern(int n, const std::vector<const SpMat*>& patterns) {
    std::vector<int> outer(static_cast<size_t>(n) + 1, 0);
    std::vector<int> inner;
    std::vector<int> rows;
    for (int column = 0; column < n; ++column) {
        rows.clear();
        for (const SpMat* pattern : patterns) {
            rows.insert(rows.end(), pattern->innerIndexPtr() + pattern->outerIndexPtr()[column],
                        pattern->innerIndexPtr() + pattern->outerIndexPtr()[column + 1]);
        }
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        inner.insert(inner.end(), rows.begin(), rows.end());
        outer[static_cast<size_t>(column) + 1] = static_cast<int>(inner.size());
    }
    const std::vector<double> values(inner.size(), 0.0);
    return Eigen::Map<const SpMat>(n, n, static_cast<Eigen::Index>(inner.size()), outer.data(), inner.data(),
                                   values.data());
}

// Where each stored entry of part sits in merged, a superset of part's pattern.
std::vector<int> positions_in(const SpMat& merged, const SpMat& part) {
    std::vector<int> positions(static_cast<size_t>(part.nonZeros()));
    for (Eigen::Index column = 0; column < part.cols(); ++column) {
        int m = merged.outerIndexPtr()[column];
        for (int k = part.outerIndexPtr()[column]; k < part.outerIndexPtr()[column + 1]; ++k) {
            while (merged.innerIndexPtr()[m] != part.innerIndexPtr()[k]) {
                ++m;
            }
            positions[static_cast<size_t>(k)] = m;
        }
    }
    return positions;
}

} // namespace

int WeightedSumEnergy::checked_size(const std::vector<WeightedEnergy>& terms) {
    if (terms.empty()) {
        throw std::invalid_argument("WeightedSumEnergy requires at least one term");
    }
    for (const WeightedEnergy& term : terms) {
        if (term.energy == nullptr) {
            throw std::invalid_argument("WeightedSumEnergy: a term has no energy");
        }
        if (term.energy->num_dofs() != terms.front().energy->num_dofs()) {
            throw std::invalid_argument("WeightedSumEnergy: all terms must have the same num_dofs");
        }
        if (!std::isfinite(term.weight) || term.weight == 0.0) {
            throw std::invalid_argument("WeightedSumEnergy: weights must be finite and nonzero");
        }
    }
    return terms.front().energy->num_dofs();
}
WeightedSumEnergy::WeightedSumEnergy(std::vector<WeightedEnergy> terms)
    : Energy(checked_size(terms)), terms_(std::move(terms)) {}

class WeightedSumEnergy::Evaluation final : public EnergyEvaluation {
  public:
    Evaluation(const WeightedSumEnergy& model, const ConstVecRef& x) : EnergyEvaluation(model, x), model_(model) {
        children_.reserve(model_.terms_.size());
        for (const auto& term : model_.terms_)
            children_.push_back(term.energy->evaluate(x));
    }

  private:
    double feasible_step_impl(const ConstVecRef& dx) const override {
        double alpha = 1.;
        for (const auto& child : children_)
            alpha = std::min(alpha, child->max_feasible_step(dx));
        return alpha;
    }
    EnergyValues compute(bool gradient, bool hessian) override {
        return assemble(gradient, hessian, false);
    }
    EnergyValues compute_projected() override {
        for (const auto& term : model_.terms_)
            if (term.weight < 0)
                return EnergyEvaluation::compute_projected();
        return assemble(true, true, true);
    }
    EnergyValues assemble(bool gradient, bool hessian, bool project) {
        ::litepgo::detail::BusyScope scope(model_.busy_, "WeightedSumEnergy");
        EnergyValues result{0, std::nullopt, std::nullopt};
        if (gradient)
            result.gradient = Vec::Zero(num_dofs());
        std::vector<const SpMat*> hessians;
        if (hessian)
            hessians.reserve(children_.size());
        for (size_t i = 0; i < children_.size(); ++i) {
            auto& child = *children_[i];
            const double w = model_.terms_[i].weight;
            if (hessian)
                hessians.push_back(&(project ? detail::EvaluationAccess::projected_hessian(child)
                                             : detail::EvaluationAccess::hessian(child)));
            if (gradient)
                *result.gradient += w * detail::EvaluationAccess::gradient(child);
            result.value += w * child.value();
        }
        if (hessian) {
            bool changed = !model_.plan_;
            for (size_t i = 0; !changed && i < hessians.size(); ++i)
                changed = !linalg::same_pattern(*hessians[i], model_.plan_->terms[i].pattern);
            if (changed) {
                AssemblyPlan next{merged_pattern(num_dofs(), hessians), {}};
                for (const auto* H : hessians)
                    next.terms.push_back({linalg::pattern_of(*H), positions_in(next.pattern, *H)});
                model_.plan_ = std::move(next);
            }
            result.hessian = model_.plan_->pattern;
            for (size_t i = 0; i < hessians.size(); ++i) {
                const double w = model_.terms_[i].weight;
                const auto& positions = model_.plan_->terms[i].positions;
                for (size_t k = 0; k < positions.size(); ++k)
                    result.hessian->valuePtr()[positions[k]] += w * hessians[i]->valuePtr()[k];
            }
        }
        return result;
    }
    const WeightedSumEnergy& model_;
    std::vector<std::shared_ptr<EnergyEvaluation>> children_;
};
std::shared_ptr<EnergyEvaluation> WeightedSumEnergy::make_evaluation(const ConstVecRef& x) const {
    return std::make_shared<Evaluation>(*this, x);
}

} // namespace litepgo::energy
