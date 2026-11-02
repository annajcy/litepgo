#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace litepgo {

using Vec = Eigen::VectorXd;
// Pass read-only vectors as const ConstVecRef&: it binds Vec, segments, and
// Maps without copying. VecRef is the writable counterpart.
using ConstVecRef = Eigen::Ref<const Eigen::VectorXd>;
using VecRef = Eigen::Ref<Eigen::VectorXd>;

// Writable non-owning outputs, including rows and blocks of row-major batches.
using StridedVecRef = Eigen::Ref<Vec, 0, Eigen::InnerStride<Eigen::Dynamic>>;
using StridedMatRef = Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

// Sparse matrices are column-major (CSC), the layout Eigen's SimplicialLDLT and
// SparseLU and Apple Accelerate factor without converting. Symmetric matrices
// store both triangles. Indices are int, matching MKL Pardiso's lp64 interface.
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
using Triplet = Eigen::Triplet<double, int>;

} // namespace litepgo
