#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/dof/dof_reduction.hpp"

#include <vector>

namespace litepgo::linalg {

// The principal submatrix of A on a DofReduction's free DOFs, refreshed in place.
//
// Construction fixes the sparsity pattern of A and of the submatrix. assign()
// then only copies values, so matrix() keeps its storage and a SparseSolver
// factoring it keeps its symbolic analysis. When A's pattern changes, build a
// new ReducedMatrix.
//
// The submatrix always stores its whole diagonal, as explicit zeros where A has
// no entry, so add_to_diagonal() can shift it without changing the pattern. With
// no fixed DOFs it is A itself plus that diagonal: in Newton it is the linear
// system's matrix either way.
class ReducedMatrix {
  public:
    // A must be compressed and reduction.num_dofs() square.
    ReducedMatrix(const dof::DofReduction& reduction, const SpMat& A);

    const SpMat& matrix() const {
        return matrix_;
    }

    // Copies the free block of A's values. A must have the construction pattern;
    // otherwise throws std::invalid_argument and leaves matrix() unchanged.
    void assign(const SpMat& A);

    // Adds shift to every diagonal entry. The next assign() overwrites it.
    void add_to_diagonal(double shift);

  private:
    // Pattern of A at construction.
    std::vector<int> outer_indices_;
    std::vector<int> inner_indices_;
    // For each stored entry of matrix_, its position in A's value array, or -1 for
    // a diagonal entry A does not store.
    std::vector<int> source_positions_;
    // Position of each diagonal entry in matrix_'s value array.
    std::vector<int> diagonal_positions_;
    SpMat matrix_;
};

} // namespace litepgo::linalg
