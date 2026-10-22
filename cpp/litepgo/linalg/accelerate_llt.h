#pragma once

#include "litepgo/linalg/sparse_solver.h"

#include <memory>

namespace litepgo {

// Apple Accelerate's sparse Cholesky. For symmetric positive definite matrices;
// an indefinite matrix fails to factorize. It reads the lower triangle.
class AccelerateLLT final : public SparseSolver {
  public:
    AccelerateLLT();
    ~AccelerateLLT() override;

  private:
    bool analyze_pattern(const SpMat& A) override;
    bool factor_values(const SpMat& A) override;
    void solve_factored(const ConstVecRef& rhs, VecRef x) const override;

    // Keeps <Eigen/AccelerateSupport>, and with it the Accelerate umbrella
    // header, out of every file that includes this one.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace litepgo
