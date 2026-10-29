#pragma once

#include "litepgo/core/types.hpp"

#include <atomic>
#include <optional>

namespace litepgo::energy {

// Which second derivative an energy assembles.
enum class HessianKind {
    // The exact Hessian.
    Exact,
    // An approximation assembled from per-element (or per-contact-pair) PSD
    // projections, for Newton steps. Each leaf energy projects its own terms;
    // composites combine the leaves' projections, so the result is PSD only when
    // theirs are and the combination keeps it (see WeightedEnergySum). Energies with
    // no natural projection, such as QuadraticEnergy, return the exact Hessian.
    Projected,
};

// A scalar energy E(x) of num_dofs() variables, with its gradient and a sparse
// Hessian whose sparsity pattern is fixed or depends on x.
//
// Using an energy:
//   - Before assembling a Hessian at x, call update_pattern(x). Values and
//     gradients never need it: they always use x's actual active terms.
//   - Hessians are written into a buffer copied from hessian_pattern(). Copy one
//     the first time, and again whenever update_pattern() returns true; eval()
//     only writes values and rejects a buffer with another pattern.
//   - One call at a time: overlapping calls throw std::logic_error.
//
// An energy holds its configuration (fixed at construction), its current pattern
// (changed only by update_pattern), and scratch buffers. It never holds x or
// cached results.
class Energy {
  public:
    virtual ~Energy() = default;
    Energy(const Energy&) = delete;
    Energy& operator=(const Energy&) = delete;

    int num_dofs() const {
        return num_dofs_;
    }
    // The current Hessian sparsity pattern, with zero values. A copy of it is a
    // Hessian buffer for eval().
    const SpMat& hessian_pattern() const {
        return pattern_;
    }
    // Whether the pattern stays the same for the energy's lifetime.
    bool has_fixed_pattern() const {
        return fixed_pattern_;
    }

    // Adapts the pattern to the state x and returns whether it changed.
    bool update_pattern(const ConstVecRef& x);

    // Returns E(x). A non-null gradient is resized to num_dofs() and filled. A
    // non-null hessian must have exactly the current pattern (std::invalid_argument
    // otherwise); only its values are written. The asymmetry is deliberate: a
    // vector's size is unambiguous, but a change of Hessian pattern must be handled
    // by the caller. The value is always finite: at an infeasible x, eval()
    // throws std::domain_error instead. If an exception escapes after the checks,
    // output values are unspecified, but the hessian's pattern never changes.
    double eval(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind = HessianKind::Exact) const;

    // A conservative alpha in [0, 1] satisfying the implemented feasibility constraints
    // along x + t * dx for every t in [0, alpha], including alpha. x must be feasible.
    // Each energy supplies its own safety margin; 1 permits the full step.
    // This bound does not guarantee representable floating-point evaluation results.
    // Newton starts at this bound; evaluation exceptions propagate without retrying.
    double max_feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const;

    double eval_value(const ConstVecRef& x) const;
    Vec eval_gradient(const ConstVecRef& x) const;
    SpMat eval_hessian(const ConstVecRef& x, HessianKind kind = HessianKind::Exact) const;

  protected:
    // pattern must be a canonical num_dofs x num_dofs matrix; its values are ignored.
    Energy(int num_dofs, const SpMat& pattern, bool fixed_pattern);

  private:
    // Hooks. refresh_pattern returns the pattern needed at x, or std::nullopt when
    // it is unchanged; update_pattern() installs it, and is the only place a pattern
    // ever changes. It is never called on an energy with a fixed pattern. A
    // composite rebuilds its tables for the new pattern before returning it, so the
    // hook is not atomic; the patterns composites build are always valid, so the
    // installation that follows cannot fail.
    // eval_values receives a gradient of size num_dofs() and a hessian with the
    // current pattern, and must overwrite every gradient entry and stored value. It
    // returns a finite value or throws std::domain_error; anything else (+-infinity,
    // NaN) is rejected as a broken energy with std::logic_error.
    virtual std::optional<SpMat> refresh_pattern(const ConstVecRef& x) = 0;
    virtual double eval_values(const ConstVecRef& x, Vec* gradient, SpMat* hessian, HessianKind kind) const = 0;
    virtual double feasible_step(const ConstVecRef& x, const ConstVecRef& dx) const;

    void check_size(const ConstVecRef& x, const char* caller) const;

    int num_dofs_;
    SpMat pattern_;
    bool fixed_pattern_;
    mutable std::atomic<bool> busy_ = false;
};

} // namespace litepgo::energy
