#include "litepgo/linalg/sparse_solver/pardiso_ldlt.hpp"

#include "litepgo/linalg/sparse_solver/pardiso_matrix.hpp"

#include <Eigen/PardisoSupport>

#include <stdexcept>

namespace litepgo::linalg {

struct PardisoLDLT::Impl {
    Impl() {
        // Eigen selects 1x1 diagonal pivoting only, which leans on pivot
        // perturbation for zero diagonals (saddle points); restore PARDISO's
        // default of Bunch-Kaufman 1x1 and 2x2 pivoting.
        solver.pardisoParameterArray()[20] = 1;
    }
    Eigen::PardisoLDLT<SpMat, Eigen::Lower> solver;
    SpMat with_diagonal;
};

PardisoLDLT::PardisoLDLT() : impl_(std::make_unique<Impl>()) {}

PardisoLDLT::~PardisoLDLT() = default;

// PARDISO can reject a pattern; Eigen then drops the analysis and would assert
// in factorize(), which SparseSolver never calls after a failed analysis.
bool PardisoLDLT::analyze_pattern(const SpMat& A) {
    impl_->solver.analyzePattern(detail::with_stored_diagonal(A, impl_->with_diagonal));
    return impl_->solver.info() == Eigen::Success;
}

bool PardisoLDLT::factor_values(const SpMat& A) {
    impl_->solver.factorize(detail::with_stored_diagonal(A, impl_->with_diagonal));
    return impl_->solver.info() == Eigen::Success;
}

void PardisoLDLT::solve_factored(const ConstVecRef& rhs, VecRef x) const {
    x = impl_->solver.solve(rhs);
    if (impl_->solver.info() != Eigen::Success) {
        throw std::runtime_error("PardisoLDLT solve failed");
    }
}

} // namespace litepgo::linalg
