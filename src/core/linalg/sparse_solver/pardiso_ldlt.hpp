#pragma once

#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#include <memory>

namespace litepgo::linalg {

// Intel oneMKL PARDISO's Bunch-Kaufman pivoted sparse LDLT, for symmetric
// indefinite matrices. It reads the lower triangle.
// PARDISO runs on TBB: the caller's task_arena and tbb::global_control bound it.
class PardisoLDLT final : public SparseSolver {
  public:
    PardisoLDLT();
    ~PardisoLDLT() override;

  private:
    bool analyze_pattern(const SpMat& A) override;
    bool factor_values(const SpMat& A) override;
    void solve_factored(const ConstVecRef& rhs, VecRef x) const override;

    // Keeps <Eigen/PardisoSupport>, and with it the MKL headers, out of every
    // file that includes this one.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace litepgo::linalg
