#pragma once
#include "litepgo/common/types.hpp"
#include <span>
#include <stdexcept>
#include <vector>

namespace litepgo::dof {
// Fixed rectangular scatter plan. Each output row is owned by one TBB task;
// contributions within a row are summed in original element order.
class ResponseAssembler final {
  public:
    using Indices = std::vector<std::vector<int>>;
    // Non-owning views: underlying buffers must outlive assemble and not alias its
    // outputs. Strides allow batch rows/blocks without per-element matrix copies.
    struct ElementView {
        using VectorView = Eigen::Map<const Vec, Eigen::Unaligned, Eigen::InnerStride<Eigen::Dynamic>>;
        using MatrixView =
            Eigen::Map<const Eigen::MatrixXd, Eigen::Unaligned, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        ElementView() = default;
        template <class V> explicit ElementView(const Eigen::MatrixBase<V>& v) : ElementView(v, Eigen::MatrixXd()) {}
        template <class V, class J>
        ElementView(const Eigen::MatrixBase<V>& v, const Eigen::MatrixBase<J>& j)
            : value_(v.derived().data()), jacobian_(j.derived().data()), size_(v.size()),
              value_stride_(v.innerStride()), rows_(j.rows()), cols_(j.cols()), row_stride_(j.rowStride()),
              col_stride_(j.colStride()) {
            static_assert(V::IsVectorAtCompileTime, "Expected a value vector");
        }
        VectorView value() const {
            return {value_, size_, Eigen::InnerStride<Eigen::Dynamic>(value_stride_)};
        }
        MatrixView jacobian() const {
            return {jacobian_, rows_, cols_, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(col_stride_, row_stride_)};
        }

      private:
        const double* value_ = nullptr;
        const double* jacobian_ = nullptr;
        Eigen::Index size_ = 0, value_stride_ = 1, rows_ = 0, cols_ = 0, row_stride_ = 1, col_stride_ = 0;
    };

    ResponseAssembler(int num_outputs, int num_inputs, Indices output_indices, Indices input_indices);
    template <class InputMap, class OutputMap>
    ResponseAssembler(const InputMap& input, const OutputMap& output)
        : ResponseAssembler(output.num_dofs(), input.num_dofs(), indices(output), indices(input)) {}
    int num_elements() const noexcept {
        return static_cast<int>(rows_.size());
    }
    std::span<const int> input_indices(int element) const {
        return columns_.at(element);
    }
    std::span<const int> output_indices(int element) const {
        return rows_.at(element);
    }
    const SpMat& jacobian_pattern() const noexcept {
        return pattern_;
    }

    // Overwrites value and requested Jacobian. One view per element in original
    // order; J must have jacobian_pattern().
    // Repeated local indices accumulate. Requested local data must be finite;
    // invalid data/dimensions/pattern throw invalid_argument before output writes.
    // Finite-input accumulation overflow throws domain_error; outputs are then unspecified.
    // Contributions use original element/local-index order, independent of threads.
    void assemble(std::span<const ElementView> elements, Vec& value, SpMat* J = nullptr) const;

  private:
    template <class Map> static Indices indices(const Map& map) {
        if (map.num_elements() < 0)
            throw std::invalid_argument("ResponseAssembler: negative element count");
        Indices result;
        result.reserve(map.num_elements());
        for (int e = 0; e < map.num_elements(); ++e)
            result.push_back(map.dof_indices(e));
        return result;
    }
    struct Contribution {
        int element, local_row;
    };
    SpMat pattern_;
    Indices rows_, columns_, offsets_;
    std::vector<std::vector<Contribution>> contributions_;
};
} // namespace litepgo::dof
