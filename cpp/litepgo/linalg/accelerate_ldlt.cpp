#include "litepgo/linalg/accelerate_ldlt.h"

#include <Eigen/AccelerateSupport>

#include <stdexcept>

namespace litepgo {

struct AccelerateLDLT::Impl {
    Eigen::AccelerateLDLT<SpMat> solver;
    // Eigen asserts in factorize() when the symbolic analysis failed, so a
    // failed analysis must turn into a failed factor() here.
    bool analyzed = false;
};

AccelerateLDLT::AccelerateLDLT() : impl_(std::make_unique<Impl>()) {}

AccelerateLDLT::~AccelerateLDLT() = default;

void AccelerateLDLT::analyze(const SpMat& A) {
    impl_->solver.analyzePattern(A);
    impl_->analyzed = impl_->solver.info() == Eigen::Success;
}

bool AccelerateLDLT::factor(const SpMat& A) {
    if (!impl_->analyzed) {
        return false;
    }
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
