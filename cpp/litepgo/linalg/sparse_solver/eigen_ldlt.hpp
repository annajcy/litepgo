#pragma once

#include "litepgo/linalg/sparse_solver/sparse_solver.hpp"

#include <Eigen/SparseCholesky>

namespace litepgo::linalg {

// Eigen's SimplicialLDLT, without pivoting. For symmetric positive definite
// matrices; it reads the lower triangle.
class EigenLDLT final : public SparseSolver {
  public:
    EigenLDLT() = default;

  private:
    bool analyze_pattern(const SpMat& A) override;
    bool factor_values(const SpMat& A) override;
    void solve_factored(const ConstVecRef& rhs, VecRef x) const override;

    Eigen::SimplicialLDLT<SpMat> solver_;
};

} // namespace litepgo::linalg
