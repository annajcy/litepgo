#pragma once
#include "litepgo/dof/response_assembler.hpp"
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include <atomic>
#include <utility>

namespace litepgo::energy {
using BatchMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Concurrent batch hooks; explicit values/Jacobians use TBB row-owned assembly.
// Hooks must be thread-safe and must not depend on batch execution order.
// Values have one row per element; Jacobians stack (output_size x input_size) blocks.
class MeshResponse : public FixedPatternResponse {
  public:
    template <class InputMap, class OutputMap>
    MeshResponse(const InputMap& input, const OutputMap& output, int batch_size = 256)
        : MeshResponse(dof::ResponseAssembler(input, output), batch_size) {}

  protected:
    virtual BatchMatrix eval_batch(std::span<const int> ids, const BatchMatrix& x, BatchMatrix* J) const = 0;
    // Defaults multiply local explicit Jacobians; overrides can avoid forming them.
    virtual BatchMatrix eval_jvp_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& dx) const;
    virtual BatchMatrix eval_vjp_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& gy) const;

    void check_batch_product(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& d,
                             bool transpose) const;

  private:
    Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& dx) override;
    Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& gy) override;
    BatchMatrix eval_batch_jacobian(std::span<const int> ids, const BatchMatrix& x) const;
    void gather(const std::vector<int>& ids, const ConstVecRef& x, BatchMatrix& result, bool output) const;
    Vec product(const ConstVecRef& x, const ConstVecRef& direction, bool transpose) const;
    MeshResponse(dof::ResponseAssembler assembler, int batch_size);
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override;
    dof::ResponseAssembler assembler_;
    struct Batch {
        std::vector<int> ids;
        BatchMatrix inputs, values, jacobian, direction;
    };
    mutable std::atomic<bool> busy_ = false;
    mutable std::vector<Batch> batches_;
    mutable std::vector<dof::ResponseAssembler::ElementView> views_;
};
} // namespace litepgo::energy
