#include "litepgo/linalg/sparse_solver/pardiso_llt.hpp"

#include "litepgo/linalg/sparse_solver/pardiso_matrix.hpp"

#include <Eigen/PardisoSupport>

#include <stdexcept>

namespace litepgo::linalg {

struct PardisoLLT::Impl {
    Eigen::PardisoLLT<SpMat, Eigen::Lower> solver;
    SpMat with_diagonal;
};

PardisoLLT::PardisoLLT() : impl_(std::make_unique<Impl>()) {}

PardisoLLT::~PardisoLLT() = default;

// PARDISO can reject a pattern; Eigen then drops the analysis and would assert
// in factorize(), which SparseSolver never calls after a failed analysis.
bool PardisoLLT::analyze_pattern(const SpMat& A) {
    impl_->solver.analyzePattern(detail::with_stored_diagonal(A, impl_->with_diagonal));
    return impl_->solver.info() == Eigen::Success;
}

bool PardisoLLT::factor_values(const SpMat& A) {
    impl_->solver.factorize(detail::with_stored_diagonal(A, impl_->with_diagonal));
    return impl_->solver.info() == Eigen::Success;
}

void PardisoLLT::solve_factored(const ConstVecRef& rhs, VecRef x) const {
    x = impl_->solver.solve(rhs);
    if (impl_->solver.info() != Eigen::Success) {
        throw std::runtime_error("PardisoLLT solve failed");
    }
}

} // namespace litepgo::linalg
