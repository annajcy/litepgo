#pragma once

#include "litepgo/common/types.hpp"

#include <span>
#include <vector>

namespace litepgo::dof {

// Splits num_dofs indices into fixed and free ones. Reduced coordinates are the
// free DOFs in increasing order. Holds indices only: fixed values belong to the
// caller, which decides what expand() leaves in the fixed entries.
class DofReduction {
  public:
    // fixed_dofs must be unique and in [0, num_dofs), in any order.
    DofReduction(int num_dofs, std::vector<int> fixed_dofs);

    int num_dofs() const {
        return num_dofs_;
    }
    int num_free() const {
        return static_cast<int>(free_dofs_.size());
    }
    std::span<const int> fixed_dofs() const {
        return fixed_dofs_;
    }
    std::span<const int> free_dofs() const {
        return free_dofs_;
    }

    // Position of dof among the free DOFs, or -1 if it is fixed.
    int free_index(int dof) const;

    // Copies the free entries of full. full and reduced must not overlap.
    void reduce(const ConstVecRef& full, VecRef reduced) const;
    Vec reduce(const ConstVecRef& full) const;

    // Writes reduced into the free entries of full, leaving the fixed entries as
    // they are: zero them first for a step, or set them to the fixed values for a
    // state. reduced and full must not overlap.
    void expand(const ConstVecRef& reduced, VecRef full) const;

    // Principal submatrix on the free DOFs, storing its whole diagonal (explicit
    // zeros where A has none). A must be compressed and num_dofs x num_dofs. Use
    // ReducedMatrix to refresh it without reallocating.
    SpMat reduce(const SpMat& A) const;

  private:
    int num_dofs_;
    std::vector<int> fixed_dofs_;
    std::vector<int> free_dofs_;
    std::vector<int> free_index_;
};

} // namespace litepgo::dof
