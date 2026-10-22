#include "litepgo/linalg/eigen_sparse_lu.h"

#include <stdexcept>

namespace litepgo {

void EigenSparseLU::analyze(const SpMat& A) {
    solver_.analyzePattern(A);
}

bool EigenSparseLU::factor(const SpMat& A) {
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
