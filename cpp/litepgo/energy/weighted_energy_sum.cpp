#include "litepgo/energy/weighted_energy_sum.hpp"

#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace litepgo {

namespace {

bool all_fixed_pattern(const std::vector<WeightedEnergy>& terms) {
    return std::all_of(terms.begin(), terms.end(),
                       [](const WeightedEnergy& term) { return term.energy->has_fixed_pattern(); });
}

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

SpMat merged_pattern(const std::vector<WeightedEnergy>& terms) {
    std::vector<const SpMat*> patterns;
    for (const WeightedEnergy& term : terms) {
        patterns.push_back(&term.energy->hessian_pattern());
    }
    return merged_pattern(terms.front().energy->num_dofs(), patterns);
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

WeightedEnergySum::Setup WeightedEnergySum::checked_setup(std::vector<WeightedEnergy> terms) {
    if (terms.empty()) {
        throw std::invalid_argument("WeightedEnergySum requires at least one term");
    }
    for (const WeightedEnergy& term : terms) {
        if (term.energy == nullptr) {
            throw std::invalid_argument("WeightedEnergySum: a term has no energy");
        }
        if (term.energy->num_dofs() != terms.front().energy->num_dofs()) {
            throw std::invalid_argument("WeightedEnergySum: all terms must have the same num_dofs");
        }
        if (!std::isfinite(term.weight) || term.weight == 0.0) {
            throw std::invalid_argument("WeightedEnergySum: weights must be finite and nonzero");
        }
    }
    SpMat pattern = merged_pattern(terms);
    const bool fixed_pattern = all_fixed_pattern(terms);
    return {std::move(terms), std::move(pattern), fixed_pattern};
}

WeightedEnergySum::WeightedEnergySum(std::vector<WeightedEnergy> terms)
    : WeightedEnergySum(checked_setup(std::move(terms))) {}

WeightedEnergySum::WeightedEnergySum(Setup setup)
    : Energy(setup.terms.front().energy->num_dofs(), setup.pattern, setup.fixed_pattern),
      terms_(std::move(setup.terms)), scratch_(terms_.size()) {
    for (size_t i = 0; i < terms_.size(); ++i) {
        scratch_[i].hessian = terms_[i].energy->hessian_pattern();
    }
    rebuild_positions(hessian_pattern());
}

std::optional<SpMat> WeightedEnergySum::refresh_pattern(const ConstVecRef& x) {
    bool changed = false;
    for (size_t i = 0; i < terms_.size(); ++i) {
        const Energy& energy = *terms_[i].energy;
        terms_[i].energy->update_pattern(x);
        // Compare patterns rather than trust the return value: the term may have
        // been updated elsewhere, in which case its own update reports no change.
        if (!same_pattern(energy.hessian_pattern(), scratch_[i].hessian)) {
            scratch_[i].hessian = energy.hessian_pattern();
            changed = true;
        }
    }
    if (!changed) {
        return std::nullopt;
    }
    std::vector<const SpMat*> patterns;
    for (const TermScratch& scratch : scratch_) {
        patterns.push_back(&scratch.hessian);
    }
    // The base installs this pattern after we return, so build positions from it here.
    SpMat merged = merged_pattern(num_dofs(), patterns);
    rebuild_positions(merged);
    return merged;
}

double WeightedEnergySum::eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const {
    if (hessian != nullptr) {
        for (size_t i = 0; i < terms_.size(); ++i) {
            if (!same_pattern(terms_[i].energy->hessian_pattern(), scratch_[i].hessian)) {
                throw std::logic_error("WeightedEnergySum: the pattern of term " + std::to_string(i) +
                                       " changed after this sum's last update_pattern(); call update_pattern() on "
                                       "the outermost energy");
            }
        }
        std::fill(hessian->valuePtr(), hessian->valuePtr() + hessian->nonZeros(), 0.0);
    }
    if (gradient != nullptr) {
        gradient->setZero();
    }

    double value = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) {
        const double weight = terms_[i].weight;
        TermScratch& scratch = scratch_[i];
        value += weight * terms_[i].energy->eval(x, gradient != nullptr ? &scratch.gradient : nullptr,
                                                 hessian != nullptr ? &scratch.hessian : nullptr, kind);
        if (gradient != nullptr) {
            *gradient += weight * scratch.gradient;
        }
        if (hessian != nullptr) {
            const double* term_values = scratch.hessian.valuePtr();
            double* values = hessian->valuePtr();
            for (size_t k = 0; k < scratch.positions.size(); ++k) {
                values[scratch.positions[k]] += weight * term_values[k];
            }
        }
    }
    return value;
}

double WeightedEnergySum::feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const {
    double alpha = 1.0;
    for (const WeightedEnergy& term : terms_) {
        alpha = std::min(alpha, term.energy->max_feasible_step(x, dx));
    }
    return alpha;
}

void WeightedEnergySum::rebuild_positions(const SpMat& pattern) {
    for (TermScratch& scratch : scratch_) {
        scratch.positions = positions_in(pattern, scratch.hessian);
    }
}

} // namespace litepgo
