#include "litepgo/linalg/sparse_solver/accelerate_ldlt.hpp"

#include <Eigen/AccelerateSupport>

#include <stdexcept>

namespace litepgo {

struct AccelerateLDLT::Impl {
    Eigen::AccelerateLDLT<SpMat> solver;
};

AccelerateLDLT::AccelerateLDLT() : impl_(std::make_unique<Impl>()) {}

AccelerateLDLT::~AccelerateLDLT() = default;

// Accelerate can reject a pattern; Eigen then drops the analysis and would assert
// in factorize(), which SparseSolver never calls after a failed analysis.
bool AccelerateLDLT::analyze_pattern(const SpMat& A) {
    impl_->solver.analyzePattern(A);
    return impl_->solver.info() == Eigen::Success;
}

bool AccelerateLDLT::factor_values(const SpMat& A) {
    impl_->solver.factorize(A);
    return impl_->solver.info() == Eigen::Success;
}

void AccelerateLDLT::solve_factored(const ConstVecRef& rhs, VecRef x) const {
    x = impl_->solver.solve(rhs);
    if (impl_->solver.info() != Eigen::Success) {
        throw std::runtime_error("AccelerateLDLT solve failed");
    }
}

} // namespace litepgo
