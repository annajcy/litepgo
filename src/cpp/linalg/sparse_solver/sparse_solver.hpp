#pragma once

#include "litepgo/common/types.hpp"

#include <atomic>
#include <vector>

namespace litepgo::linalg {

// Direct factorization of a square sparse matrix, split into its two phases:
//
//   analyze(A)    symbolic analysis of A's sparsity pattern (ordering, structure of
//                 the factors); depends on where the nonzeros are, not their values.
//   factorize(A)  numeric factorization of a matrix with the analyzed pattern.
//
// The caller decides when the pattern changed and analyzes again; the solver never
// re-analyzes on its own. It keeps the analyzed pattern only to reject a factorize()
// whose matrix does not match it, since backends would otherwise compute garbage or
// corrupt memory.
//
// Matrices must be compressed, with strictly increasing row indices in every column
// (sorted, no duplicate entries). Symmetric matrices store both triangles. Symmetry
// and definiteness are caller contracts and are not checked.
//
// One solver serves one caller at a time: concurrent calls throw std::logic_error.
class SparseSolver {
  public:
    virtual ~SparseSolver() = default;
    SparseSolver(const SparseSolver&) = delete;
    SparseSolver& operator=(const SparseSolver&) = delete;

    // Symbolic analysis of A's pattern; discards any factorization. Returns false if
    // the backend cannot analyze it. Invalid input throws std::invalid_argument.
    bool analyze(const SpMat& A);

    // Numeric factorization. A must have exactly the pattern of the last successful
    // analyze(), or std::invalid_argument is thrown; without one, std::logic_error.
    // Returns false on numerical failure, which keeps the analysis, so the caller can
    // retry with other values (a diagonal shift, say) on the same pattern.
    bool factorize(const SpMat& A);

    // analyze(A), then factorize(A): for a matrix factored once.
    bool compute(const SpMat& A);

    // Solves with the current factorization. rhs and x must not overlap.
    // Throws std::logic_error without a factorization, and std::runtime_error if the
    // backend fails.
    void solve(const ConstVecRef& rhs, VecRef x) const;
    Vec solve(const ConstVecRef& rhs) const;

  protected:
    SparseSolver() = default;

  private:
    // Backend hooks. factor_values is only called after analyze_pattern succeeded on
    // the same pattern, and solve_factored only after factor_values succeeded.
    virtual bool analyze_pattern(const SpMat& A) = 0;
    virtual bool factor_values(const SpMat& A) = 0;
    virtual void solve_factored(const ConstVecRef& rhs, VecRef x) const = 0;

    bool matches_analyzed_pattern(const SpMat& A) const;
    void forget_analysis();

    // Pattern of the last successful analyze(); used only to validate factorize().
    std::vector<int> outer_indices_;
    std::vector<int> inner_indices_;
    bool analyzed_ = false;
    bool factored_ = false;
    mutable std::atomic<bool> busy_ = false;
};

} // namespace litepgo::linalg
