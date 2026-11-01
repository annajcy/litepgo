#pragma once
#include "litepgo/dof/response_assembler.hpp"
#include "litepgo/response/fixed_pattern_response.hpp"
#include <functional>
#include <utility>

namespace litepgo::response {
using BatchMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Concurrent batch callbacks followed by TBB row-owned assembly in element order.
// Callbacks must be thread-safe and must not depend on batch execution order.
// Values have one row per element; Jacobians stack (output_size x input_size) blocks.
class MeshResponse final : public FixedPatternResponse {
  public:
    using Callback = std::function<BatchMatrix(std::span<const int>, const BatchMatrix&, BatchMatrix*)>;
    template <class InputMap, class OutputMap>
    MeshResponse(const InputMap& input, const OutputMap& output, Callback callback, int batch_size = 256)
        : MeshResponse(dof::ResponseAssembler(input, output), std::move(callback), batch_size) {}

  private:
    MeshResponse(dof::ResponseAssembler assembler, Callback callback, int batch_size);
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override;
    dof::ResponseAssembler assembler_;
    Callback callback_;
    std::vector<std::vector<int>> batches_;
    struct Storage {
        BatchMatrix inputs, values, jacobian;
    };
    mutable std::vector<Storage> storage_;
    mutable std::vector<dof::ResponseAssembler::ElementView> views_;
};
} // namespace litepgo::response
