#include "litepgo/linalg/sparse_solver/pardiso_llt.hpp"

#include <Eigen/PardisoSupport>

#include <stdexcept>

namespace litepgo::linalg {

struct PardisoLLT::Impl {
    Eigen::PardisoLLT<SpMat, Eigen::Lower> solver;
};

PardisoLLT::PardisoLLT() : impl_(std::make_unique<Impl>()) {}

PardisoLLT::~PardisoLLT() = default;

// PARDISO can reject a pattern; Eigen then drops the analysis and would assert
// in factorize(), which SparseSolver never calls after a failed analysis.
bool PardisoLLT::analyze_pattern(const SpMat& A) {
    impl_->solver.analyzePattern(A);
    return impl_->solver.info() == Eigen::Success;
}

bool PardisoLLT::factor_values(const SpMat& A) {
    impl_->solver.factorize(A);
    return impl_->solver.info() == Eigen::Success;
}

void PardisoLLT::solve_factored(const ConstVecRef& rhs, VecRef x) const {
    x = impl_->solver.solve(rhs);
    if (impl_->solver.info() != Eigen::Success) {
        throw std::runtime_error("PardisoLLT solve failed");
    }
}

} // namespace litepgo::linalg
