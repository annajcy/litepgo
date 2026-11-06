#include "litepgo/energy/scalar/weighted_sum_energy.hpp"

#include "litepgo/common/busy_scope.hpp"
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

EnergyEvaluation WeightedSumEnergy::eval_impl(const ConstVecRef& x, bool gradient,
                                              std::optional<HessianKind> kind) const {
    ::litepgo::detail::BusyScope scope(busy_, "WeightedSumEnergy");
    EnergyEvaluation result{0, std::nullopt, std::nullopt};
    if (gradient)
        result.gradient = Vec::Zero(num_dofs());
    std::vector<EnergyEvaluation> children;
    children.reserve(terms_.size());
    for (const auto& term : terms_)
        children.push_back(term.energy->eval(x, gradient, kind));
    if (kind) {
        bool changed = !plan_;
        for (size_t i = 0; !changed && i < children.size(); ++i)
            changed = !linalg::same_pattern(*children[i].hessian, plan_->terms[i].pattern);
        if (changed) {
            std::vector<const SpMat*> patterns;
            for (const auto& child : children)
                patterns.push_back(&*child.hessian);
            AssemblyPlan next{merged_pattern(num_dofs(), patterns), {}};
            for (const auto& child : children)
                next.terms.push_back({linalg::pattern_of(*child.hessian), positions_in(next.pattern, *child.hessian)});
            plan_ = std::move(next);
        }
        result.hessian = plan_->pattern;
    }
    for (size_t i = 0; i < terms_.size(); ++i) {
        const double w = terms_[i].weight;
        result.value += w * children[i].value;
        if (gradient)
            *result.gradient += w * *children[i].gradient;
        if (kind) {
            const auto& positions = plan_->terms[i].positions;
            for (size_t k = 0; k < positions.size(); ++k)
                result.hessian->valuePtr()[positions[k]] += w * children[i].hessian->valuePtr()[k];
        }
    }
    return result;
}
double WeightedSumEnergy::feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    double alpha = 1.0;
    for (const auto& term : terms_)
        alpha = std::min(alpha, term.energy->max_feasible_step(x, dx));
    return alpha;
}
} // namespace litepgo::energy
