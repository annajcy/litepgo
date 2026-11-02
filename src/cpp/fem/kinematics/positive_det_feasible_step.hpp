#pragma once

#include <Eigen/Core>

namespace litepgo::fem {

// Safe inclusive bound in [0,1] for det(A+t*delta_A)>0 over the whole segment.
// Finds the first zero of the determinant cubic, including repeated roots, and
// keeps 10% of the distance to it. Returns 1 if no zero is approached on [0,1].
// Non-finite inputs throw invalid_argument; a non-finite/non-positive initial
// determinant throws std::domain_error. An unrepresentable direction may return 0.
double eval_positive_det_feasible_step(const Eigen::Matrix3d& A, const Eigen::Matrix3d& delta_A);

} // namespace litepgo::fem
