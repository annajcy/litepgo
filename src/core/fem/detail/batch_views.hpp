#pragma once
#include "litepgo/common/types.hpp"
#include <optional>

namespace litepgo::fem::detail {
// Batch rows hold vectors; consecutive groups of rows hold matrices.
template <class Matrix> std::optional<StridedVecRef> batch_vector_view(Matrix* output, Eigen::Index batch) {
    if (!output)
        return std::nullopt;
    return StridedVecRef(output->row(batch).transpose());
}
template <class Matrix>
std::optional<StridedMatRef> batch_matrix_view(Matrix* output, Eigen::Index batch, Eigen::Index rows) {
    if (!output)
        return std::nullopt;
    static_assert(Matrix::IsRowMajor);
    Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> block(
        output->data() + batch * rows * output->outerStride(), rows, output->cols(),
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(output->innerStride(), output->outerStride()));
    return StridedMatRef(block);
}
} // namespace litepgo::fem::detail
