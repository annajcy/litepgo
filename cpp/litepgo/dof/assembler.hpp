#pragma once

#include "litepgo/core/types.hpp"
#include <span>
#include <vector>

namespace litepgo::dof {

// Fixed scatter plan. Local entries follow the supplied per-element index order.
// Outputs must be initialized by the caller. Adds values without resizing or
// changing sparse structure. Calls writing the same output must be serialized.
class Assembler final {
  public:
    Assembler(int num_dofs, std::vector<std::vector<int>> element_dof_indices);

    std::span<const int> dof_indices(int element) const;
    int num_elements() const noexcept;

    const SpMat& hessian_pattern() const noexcept;

    // Inputs must not alias the output. Invalid indices, sizes, non-finite local
    // values, or a mismatched sparse pattern are rejected before any writes.
    void add_gradient(int element, const ConstVecRef& local, VecRef global) const;
    void add_hessian(int element, const Eigen::Ref<const Eigen::MatrixXd>& local, SpMat& global) const;

    // Overwrites gradient (resized to num_dofs) and the values of hessian with the sums
    // of one local result per element, in parallel and in a fixed order. Each requested
    // span holds num_elements() results sized to their elements' DOFs; hessian must have
    // hessian_pattern(). Invalid sizes or patterns throw std::invalid_argument before any
    // write; a non-finite sum throws std::domain_error and leaves the outputs unspecified.
    void assemble(std::span<const Vec> gradients, std::span<const Eigen::MatrixXd> hessians, Vec* gradient,
                  SpMat* hessian) const;

  private:
    void check_element(int element) const;

    struct Contribution {
        int element;
        int local_dof;
    };

    int num_dofs_;
    std::vector<std::vector<int>> element_dof_indices_;
    SpMat pattern_;
    std::vector<std::vector<int>> hessian_offsets_;
    std::vector<std::vector<Contribution>> dof_contributions_;
};

} // namespace litepgo::dof
