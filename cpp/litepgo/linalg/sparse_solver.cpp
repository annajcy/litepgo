#include "litepgo/linalg/sparse_solver.h"

#include <algorithm>
#include <stdexcept>

namespace litepgo {

namespace {

// Marks a solver busy for one call, so overlapping calls fail loudly.
class BusyScope {
  public:
    explicit BusyScope(std::atomic<bool>& busy) : busy_(busy) {
        if (busy_.exchange(true)) {
            throw std::logic_error("SparseSolver is already in use by another call");
        }
    }
    ~BusyScope() {
        busy_.store(false);
    }
    BusyScope(const BusyScope&) = delete;
    BusyScope& operator=(const BusyScope&) = delete;

  private:
    std::atomic<bool>& busy_;
};

void check_matrix(const SpMat& A) {
    if (A.rows() == 0 || A.rows() != A.cols()) {
        throw std::invalid_argument("SparseSolver requires a non-empty square matrix");
    }
    if (!A.isCompressed()) {
        throw std::invalid_argument("SparseSolver requires a compressed matrix; call makeCompressed()");
    }
    const int* outer = A.outerIndexPtr();
    const int* inner = A.innerIndexPtr();
    for (Eigen::Index column = 0; column < A.cols(); ++column) {
        for (int k = outer[column] + 1; k < outer[column + 1]; ++k) {
            if (inner[k - 1] >= inner[k]) {
                throw std::invalid_argument("SparseSolver requires strictly increasing row indices in each column; "
                                            "sum duplicate entries first (scipy: sum_duplicates())");
            }
        }
    }
}

bool overlaps(const double* a, const double* b, Eigen::Index size) {
    return size > 0 && a < b + size && b < a + size;
}

} // namespace

bool SparseSolver::factorize(const SpMat& A) {
    BusyScope scope(busy_);
    check_matrix(A);
    factored_ = false;
    try {
        if (!same_pattern(A)) {
            outer_indices_.clear();
            inner_indices_.clear();
            analyze(A);
            remember_pattern(A);
        }
        factored_ = factor(A);
    } catch (...) {
        outer_indices_.clear();
        inner_indices_.clear();
        throw;
    }
    if (!factored_) {
        // Analyze again next time instead of trusting state a failed backend left behind.
        outer_indices_.clear();
        inner_indices_.clear();
    }
    return factored_;
}

void SparseSolver::solve(const ConstVecRef& rhs, VecRef x) const {
    BusyScope scope(busy_);
    if (!factored_) {
        throw std::logic_error("SparseSolver::solve requires a successful factorize()");
    }
    const auto size = static_cast<Eigen::Index>(outer_indices_.size()) - 1;
    if (rhs.size() != size || x.size() != size) {
        throw std::invalid_argument("SparseSolver::solve vector sizes must match the factorized matrix");
    }
    if (overlaps(rhs.data(), x.data(), size)) {
        throw std::invalid_argument("SparseSolver::solve requires rhs and x not to overlap");
    }
    solve_factored(rhs, x);
}

Vec SparseSolver::solve(const ConstVecRef& rhs) const {
    Vec x(rhs.size());
    solve(rhs, x);
    return x;
}

bool SparseSolver::same_pattern(const SpMat& A) const {
    return !outer_indices_.empty() && outer_indices_.size() == static_cast<size_t>(A.outerSize() + 1) &&
           inner_indices_.size() == static_cast<size_t>(A.nonZeros()) &&
           std::equal(outer_indices_.begin(), outer_indices_.end(), A.outerIndexPtr()) &&
           std::equal(inner_indices_.begin(), inner_indices_.end(), A.innerIndexPtr());
}

void SparseSolver::remember_pattern(const SpMat& A) {
    outer_indices_.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
    inner_indices_.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
}

} // namespace litepgo
