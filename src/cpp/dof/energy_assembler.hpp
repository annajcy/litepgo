#pragma once

#include "litepgo/common/types.hpp"
#include <span>
#include <vector>

namespace litepgo::dof {

// Fixed scatter plan. Local entries follow the supplied per-element index order.
// assemble overwrites outputs; add_gradient/add_hessian add to initialized outputs.
// Sparse structure is preserved. Calls writing the same output must be serialized.
class EnergyAssembler final {
  public:
    // Borrowed derivatives, including rows/blocks of batch matrices. Buffers must
    // outlive assemble and must not alias its outputs. Unrequested views may be empty.
    struct ElementView {
        using VectorView = Eigen::Map<const Vec, Eigen::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;
        using MatrixView =
            Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        ElementView() = default;
        template <class G, class H>
        ElementView(const Eigen::MatrixBase<G>& g, const Eigen::MatrixBase<H>& h)
            : gradient_(g.derived().data()), hessian_(h.derived().data()), size_(g.size()),
              gradient_stride_(g.innerStride()), rows_(h.rows()), cols_(h.cols()), row_stride_(h.rowStride()),
              col_stride_(h.colStride()) {
            static_assert(G::IsVectorAtCompileTime, "Expected a gradient vector");
        }
        VectorView gradient() const {
            return {gradient_, size_, Eigen::InnerStride<Eigen::Dynamic>(gradient_stride_)};
        }
        MatrixView hessian() const {
            return {hessian_, rows_, cols_, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(col_stride_, row_stride_)};
        }

      private:
        const double* gradient_ = nullptr;
        const double* hessian_ = nullptr;
        Eigen::Index size_ = 0, gradient_stride_ = 1, rows_ = 0, cols_ = 0, row_stride_ = 1, col_stride_ = 0;
    };

    // Repeated local indices accumulate. Requested local data must be finite;
    // invalid data/dimensions/pattern throw invalid_argument before output writes.
    // Finite-input accumulation overflow throws domain_error; outputs are then unspecified.
    // Contributions use original element/local-index order, independent of threads.
    void assemble(std::span<const ElementView> elements, Vec* gradient, SpMat* hessian) const;

    EnergyAssembler(int num_dofs, std::vector<std::vector<int>> element_dof_indices);

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
