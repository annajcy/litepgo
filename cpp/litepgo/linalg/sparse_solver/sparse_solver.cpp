#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#include "litepgo/common/busy_scope.hpp"
#include "litepgo/common/memory.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"

#include <algorithm>
#include <stdexcept>

namespace litepgo::linalg {

namespace {

void check_matrix(const SpMat& A) {
    if (A.rows() == 0 || A.rows() != A.cols()) {
        throw std::invalid_argument("SparseSolver requires a non-empty square matrix");
    }
    if (!A.isCompressed()) {
        throw std::invalid_argument("SparseSolver requires a compressed matrix; call makeCompressed()");
    }
    if (!is_canonical(A)) {
        throw std::invalid_argument("SparseSolver requires strictly increasing row indices in each column; "
                                    "sum duplicate entries first");
    }
}

} // namespace

bool SparseSolver::analyze(const SpMat& A) {
    ::litepgo::detail::BusyScope scope(busy_, "SparseSolver");
    check_matrix(A);
    forget_analysis();
    // A hook that throws leaves the solver unanalyzed, as forget_analysis() set it.
    if (!analyze_pattern(A)) {
        return false;
    }
    outer_indices_.assign(A.outerIndexPtr(), A.outerIndexPtr() + A.outerSize() + 1);
    inner_indices_.assign(A.innerIndexPtr(), A.innerIndexPtr() + A.nonZeros());
    analyzed_ = true;
    return true;
}

bool SparseSolver::factorize(const SpMat& A) {
    ::litepgo::detail::BusyScope scope(busy_, "SparseSolver");
    if (!analyzed_) {
        throw std::logic_error("SparseSolver::factorize requires a successful analyze()");
    }
    if (!matches_analyzed_pattern(A)) {
        throw std::invalid_argument("SparseSolver::factorize: the matrix does not have the analyzed sparsity pattern; "
                                    "call analyze() for the new pattern");
    }
    factored_ = false;
    try {
        factored_ = factor_values(A);
    } catch (...) {
        // Do not trust a backend that failed unexpectedly: start over from analyze().
        forget_analysis();
        throw;
    }
    return factored_;
}

bool SparseSolver::compute(const SpMat& A) {
    return analyze(A) && factorize(A);
}

void SparseSolver::solve(const ConstVecRef& rhs, VecRef x) const {
    ::litepgo::detail::BusyScope scope(busy_, "SparseSolver");
    if (!factored_) {
        throw std::logic_error("SparseSolver::solve requires a successful factorize()");
    }
    const auto size = static_cast<Eigen::Index>(outer_indices_.size()) - 1;
    if (rhs.size() != size || x.size() != size) {
        throw std::invalid_argument("SparseSolver::solve vector sizes must match the factorized matrix");
    }
    if (::litepgo::detail::overlaps(rhs.data(), size, x.data(), size)) {
        throw std::invalid_argument("SparseSolver::solve requires rhs and x not to overlap");
    }
    solve_factored(rhs, x);
}

Vec SparseSolver::solve(const ConstVecRef& rhs) const {
    Vec x(rhs.size());
    solve(rhs, x);
    return x;
}

bool SparseSolver::matches_analyzed_pattern(const SpMat& A) const {
    return A.isCompressed() && A.rows() == static_cast<Eigen::Index>(outer_indices_.size()) - 1 &&
           outer_indices_.size() == static_cast<size_t>(A.outerSize() + 1) &&
           inner_indices_.size() == static_cast<size_t>(A.nonZeros()) &&
           std::equal(outer_indices_.begin(), outer_indices_.end(), A.outerIndexPtr()) &&
           std::equal(inner_indices_.begin(), inner_indices_.end(), A.innerIndexPtr());
}

void SparseSolver::forget_analysis() {
    analyzed_ = false;
    factored_ = false;
    outer_indices_.clear();
    inner_indices_.clear();
}

} // namespace litepgo::linalg
