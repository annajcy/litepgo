#include "litepgo/linalg/eigen_ldlt.h"

#include <stdexcept>

namespace litepgo {

void EigenLDLT::analyze(const SpMat& A) {
    solver_.analyzePattern(A);
}

bool EigenLDLT::factor(const SpMat& A) {
    solver_.factorize(A);
    return solver_.info() == Eigen::Success;
}

void EigenLDLT::solve_factored(const ConstVecRef& rhs, VecRef x) const {
    x = solver_.solve(rhs);
    if (solver_.info() != Eigen::Success) {
        throw std::runtime_error("EigenLDLT solve failed");
    }
}

} // namespace litepgo
