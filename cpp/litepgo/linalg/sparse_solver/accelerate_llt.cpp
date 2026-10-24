#include "litepgo/linalg/sparse_solver/accelerate_llt.hpp"

#include <Eigen/AccelerateSupport>

#include <stdexcept>

namespace litepgo {

struct AccelerateLLT::Impl {
    Eigen::AccelerateLLT<SpMat> solver;
};

AccelerateLLT::AccelerateLLT() : impl_(std::make_unique<Impl>()) {}

AccelerateLLT::~AccelerateLLT() = default;

// Accelerate can reject a pattern; Eigen then drops the analysis and would assert
// in factorize(), which SparseSolver never calls after a failed analysis.
bool AccelerateLLT::analyze_pattern(const SpMat& A) {
    impl_->solver.analyzePattern(A);
    return impl_->solver.info() == Eigen::Success;
}

bool AccelerateLLT::factor_values(const SpMat& A) {
    impl_->solver.factorize(A);
    return impl_->solver.info() == Eigen::Success;
}

void AccelerateLLT::solve_factored(const ConstVecRef& rhs, VecRef x) const {
    x = impl_->solver.solve(rhs);
    if (impl_->solver.info() != Eigen::Success) {
        throw std::runtime_error("AccelerateLLT solve failed");
    }
}

} // namespace litepgo
