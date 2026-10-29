#pragma once

#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseLU>

namespace litepgo::linalg {

// Eigen's SparseLU with partial pivoting. Handles symmetric indefinite (and
// general nonsingular) matrices, at a higher cost than an LDLT.
class EigenSparseLU final : public SparseSolver {
  public:
    EigenSparseLU() = default;

  private:
    bool analyze_pattern(const SpMat& A) override;
    bool factor_values(const SpMat& A) override;
    void solve_factored(const ConstVecRef& rhs, VecRef x) const override;

    Eigen::SparseLU<SpMat, Eigen::COLAMDOrdering<int>> solver_;
};

} // namespace litepgo::linalg
