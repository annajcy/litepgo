#pragma once

#include "litepgo/core/types.h"

#include <atomic>
#include <vector>

namespace litepgo {

// Factorization of a square sparse matrix, and solves against it.
//
// Input matrices must be compressed, with strictly increasing row indices in
// every column (sorted, no duplicate entries). Symmetric matrices store both
// triangles. Symmetry and definiteness are caller contracts and are not checked.
//
// The symbolic analysis is reused while the sparsity pattern stays the same, so
// refactorizing a matrix with new values only redoes the numeric factorization.
//
// One solver serves one caller at a time: concurrent factorize or solve calls
// throw std::logic_error instead of racing.
class SparseSolver {
  public:
    virtual ~SparseSolver() = default;
    SparseSolver(const SparseSolver&) = delete;
    SparseSolver& operator=(const SparseSolver&) = delete;

    // Returns false on numerical failure, which leaves no usable factorization.
    // Invalid input throws std::invalid_argument.
    bool factorize(const SpMat& A);

    // Solves with the last successful factorization. rhs and x must not overlap.
    // Throws std::logic_error without a factorization, and std::runtime_error if
    // the backend fails.
    void solve(const ConstVecRef& rhs, VecRef x) const;
    Vec solve(const ConstVecRef& rhs) const;

  protected:
    SparseSolver() = default;

  private:
    // Backend hooks. analyze always precedes factor for a new pattern, and
    // solve_factored is only called after factor succeeded.
    virtual void analyze(const SpMat& A) = 0;
    virtual bool factor(const SpMat& A) = 0;
    virtual void solve_factored(const ConstVecRef& rhs, VecRef x) const = 0;

    bool same_pattern(const SpMat& A) const;
    void remember_pattern(const SpMat& A);

    // Pattern of the last analysis; empty when the next factorize must analyze.
    std::vector<int> outer_indices_;
    std::vector<int> inner_indices_;
    bool factored_ = false;
    mutable std::atomic<bool> busy_ = false;
};

} // namespace litepgo
