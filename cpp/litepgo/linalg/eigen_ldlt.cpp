#include "litepgo/linalg/eigen_ldlt.h"

#include <stdexcept>

namespace litepgo {

bool EigenLDLT::analyze_pattern(const SpMat& A) {
    // Eigen's analysis has no failure mode short of throwing on allocation.
    solver_.analyzePattern(A);
    return true;
}

bool EigenLDLT::factor_values(const SpMat& A) {
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
