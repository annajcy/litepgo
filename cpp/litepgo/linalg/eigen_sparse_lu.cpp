#include "litepgo/linalg/eigen_sparse_lu.h"

#include <stdexcept>

namespace litepgo {

bool EigenSparseLU::analyze_pattern(const SpMat& A) {
    // Eigen's analysis has no failure mode short of throwing on allocation.
    solver_.analyzePattern(A);
    return true;
}

bool EigenSparseLU::factor_values(const SpMat& A) {
    solver_.factorize(A);
    return solver_.info() == Eigen::Success;
}

void EigenSparseLU::solve_factored(const ConstVecRef& rhs, VecRef x) const {
    x = solver_.solve(rhs);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("EigenSparseLU solve failed");
    }
}

} // namespace litepgo
