#pragma once

// Small energies for testing the energy framework and its composites.

#include "litepgo/energy/scalar/dynamic_pattern_energy.hpp"
#include "litepgo/energy/scalar/fixed_pattern_energy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace litepgo::energy::test {

// Whether pattern stores (row, column).
inline bool stores(const SpMat& pattern, int row, int column) {
    const int* begin = pattern.innerIndexPtr() + pattern.outerIndexPtr()[column];
    const int* end = pattern.innerIndexPtr() + pattern.outerIndexPtr()[column + 1];
    return std::binary_search(begin, end, row);
}

// Symmetric pattern holding the diagonal and the given (i, i + 1) pairs.
inline SpMat chain_pattern(int n, const std::vector<int>& pair_starts) {
    std::vector<Triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 0.0);
    }
    for (const int i : pair_starts) {
        entries.emplace_back(i, i + 1, 0.0);
        entries.emplace_back(i + 1, i, 0.0);
    }
    SpMat pattern(n, n);
    pattern.setFromTriplets(entries.begin(), entries.end());
    return pattern;
}

inline std::vector<int> all_pair_starts(int n) {
    std::vector<int> starts;
    for (int i = 0; i + 1 < n; ++i) {
        starts.push_back(i);
    }
    return starts;
}

inline void zero_values(SpMat& matrix) {
    std::fill(matrix.valuePtr(), matrix.valuePtr() + matrix.nonZeros(), 0.0);
}

// E(x) = sum_i (x_i^2 - 1)^2 + sum_i (x_i - x_{i+1})^2: nonlinear, tridiagonal.
// Its projected Hessian clamps each double well's curvature 12 x^2 - 4 at zero,
// so it differs from the exact one wherever |x_i| < 1 / sqrt(3).
class DoubleWellChain final : public energy::FixedPatternEnergy {
  public:
    explicit DoubleWellChain(int n) : energy::FixedPatternEnergy(n, chain_pattern(n, all_pair_starts(n))) {}

  private:
    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, energy::HessianKind kind) const override {
        const int n = num_dofs();
        double value = 0.0;
        if (gradient != nullptr) {
            gradient->setZero();
        }
        if (hessian != nullptr) {
            zero_values(*hessian);
        }
        for (int i = 0; i < n; ++i) {
            const double well = x[i] * x[i] - 1.0;
            value += well * well;
            if (gradient != nullptr) {
                (*gradient)[i] += 4.0 * x[i] * well;
            }
            if (hessian != nullptr) {
                const double curvature = 12.0 * x[i] * x[i] - 4.0;
                hessian->coeffRef(i, i) +=
                    kind == energy::HessianKind::Projected ? std::max(curvature, 0.0) : curvature;
            }
        }
        for (int i = 0; i + 1 < n; ++i) {
            const double difference = x[i] - x[i + 1];
            value += difference * difference;
            if (gradient != nullptr) {
                (*gradient)[i] += 2.0 * difference;
                (*gradient)[i + 1] -= 2.0 * difference;
            }
            if (hessian != nullptr) {
                hessian->coeffRef(i, i) += 2.0;
                hessian->coeffRef(i + 1, i + 1) += 2.0;
                hessian->coeffRef(i, i + 1) -= 2.0;
                hessian->coeffRef(i + 1, i) -= 2.0;
            }
        }
        return value;
    }
};

// E(x) = -log(1 - x_k), defined for x_k < 1; elsewhere it throws std::domain_error.
// Its feasible step keeps 10% of the remaining distance to the boundary.
class HalfSpaceBarrier final : public energy::FixedPatternEnergy {
  public:
    HalfSpaceBarrier(int n, int k) : energy::FixedPatternEnergy(n, single_entry_pattern(n, k)), k_(k) {}

  private:
    static SpMat single_entry_pattern(int n, int k) {
        SpMat pattern(n, n);
        pattern.insert(k, k) = 0.0;
        pattern.makeCompressed();
        return pattern;
    }

    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, energy::HessianKind) const override {
        const double gap = 1.0 - x[k_];
        if (!(gap > 0.0)) {
            throw std::domain_error("HalfSpaceBarrier: x_k must be below 1");
        }
        if (gradient != nullptr) {
            gradient->setZero();
        }
        if (hessian != nullptr) {
            zero_values(*hessian);
        }
        if (gradient != nullptr) {
            (*gradient)[k_] = 1.0 / gap;
        }
        if (hessian != nullptr) {
            hessian->coeffRef(k_, k_) = 1.0 / (gap * gap);
        }
        return -std::log(gap);
    }

    double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const override {
        if (dx[k_] <= 0.0) {
            return 1.0;
        }
        return std::min(1.0, 0.9 * (1.0 - x[k_]) / dx[k_]);
    }

    int k_;
};

// A dynamic-pattern energy: E(x) = sum_i x_i^2 + sum over active pairs of
// x_i x_{i+1}, where (i, i + 1) is active when x_i + x_{i+1} > 0. Its pattern
// holds the diagonal and the active pairs, like contact holding active contacts.
class SwitchingEnergy final : public energy::DynamicPatternEnergy {
  public:
    explicit SwitchingEnergy(int n) : energy::DynamicPatternEnergy(n, chain_pattern(n, {})) {}

    static std::vector<int> active_pairs(const ConstVecRef& x) {
        std::vector<int> active;
        for (int i = 0; i + 1 < x.size(); ++i) {
            if (x[i] + x[i + 1] > 0.0) {
                active.push_back(i);
            }
        }
        return active;
    }

  private:
    SpMat pattern_at(const ConstVecRef& x) const override {
        return chain_pattern(num_dofs(), active_pairs(x));
    }

    double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, energy::HessianKind) const override {
        const std::vector<int> active = active_pairs(x);
        if (hessian != nullptr) {
            for (const int i : active) {
                if (!stores(hessian_pattern(), i + 1, i)) {
                    throw std::logic_error("SwitchingEnergy: an active pair is outside the pattern; call "
                                           "update_pattern() at this state");
                }
            }
            zero_values(*hessian);
        }
        double value = x.squaredNorm();
        if (gradient != nullptr) {
            *gradient = 2.0 * x;
        }
        if (hessian != nullptr) {
            for (int i = 0; i < num_dofs(); ++i) {
                hessian->coeffRef(i, i) = 2.0;
            }
        }
        for (const int i : active) {
            value += x[i] * x[i + 1];
            if (gradient != nullptr) {
                (*gradient)[i] += x[i + 1];
                (*gradient)[i + 1] += x[i];
            }
            if (hessian != nullptr) {
                hessian->coeffRef(i, i + 1) = 1.0;
                hessian->coeffRef(i + 1, i) = 1.0;
            }
        }
        return value;
    }
};

} // namespace litepgo::energy::test
