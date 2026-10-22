#include "litepgo/linalg/accelerate_llt.h"

#include <Eigen/AccelerateSupport>

#include <stdexcept>

namespace litepgo {

struct AccelerateLLT::Impl {
    Eigen::AccelerateLLT<SpMat> solver;
    // Eigen asserts in factorize() when the symbolic analysis failed, so a
    // failed analysis must turn into a failed factor() here.
    bool analyzed = false;
};

AccelerateLLT::AccelerateLLT() : impl_(std::make_unique<Impl>()) {}

AccelerateLLT::~AccelerateLLT() = default;

void AccelerateLLT::analyze(const SpMat& A) {
    impl_->solver.analyzePattern(A);
    impl_->analyzed = impl_->solver.info() == Eigen::Success;
}

bool AccelerateLLT::factor(const SpMat& A) {
    if (!impl_->analyzed) {
        return false;
    }
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
