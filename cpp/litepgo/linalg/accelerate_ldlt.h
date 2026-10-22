#pragma once

#include "litepgo/linalg/sparse_solver.h"

#include <memory>

namespace litepgo {

// Apple Accelerate's pivoted sparse LDLT, for symmetric indefinite matrices.
// It reads the lower triangle.
class AccelerateLDLT final : public SparseSolver {
  public:
    AccelerateLDLT();
    ~AccelerateLDLT() override;

  private:
    void analyze(const SpMat& A) override;
    bool factor(const SpMat& A) override;
    void solve_factored(const ConstVecRef& rhs, VecRef x) const override;

    // Keeps <Eigen/AccelerateSupport>, and with it the Accelerate umbrella
    // header, out of every file that includes this one.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace litepgo
