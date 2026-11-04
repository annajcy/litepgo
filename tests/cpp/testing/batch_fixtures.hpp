#pragma once
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include <functional>
namespace litepgo::testing {
using energy::BatchMatrix;
// Test-only injection of malformed and reference formulas into the subclass protocol.
class BatchEnergyFixture : public energy::MeshEnergy {
  public:
    using Callback = std::function<Vec(std::span<const int>, const BatchMatrix&, BatchMatrix*, BatchMatrix*)>;
    using Step = std::function<Vec(std::span<const int>, const BatchMatrix&, const BatchMatrix&)>;
    template <class Map>
    BatchEnergyFixture(const Map& map, Callback callback, int batch_size = 256, Step step = {})
        : energy::MeshEnergy(map, batch_size), callback_(std::move(callback)), step_(std::move(step)) {
        if (!callback_)
            throw std::invalid_argument("BatchEnergyFixture: empty callback");
    }

  protected:
    Vec eval_batch(std::span<const int> ids, const BatchMatrix& x, BatchMatrix* g, BatchMatrix* H) const override {
        return callback_(ids, x, g, H);
    }
    Vec eval_feasible_step_batch(std::span<const int> ids, const BatchMatrix& x, const BatchMatrix& dx) const override {
        return step_ ? step_(ids, x, dx) : energy::MeshEnergy::eval_feasible_step_batch(ids, x, dx);
    }

  private:
    Callback callback_;
    Step step_;
};

class BatchResponseFixture : public energy::MeshResponse {
  public:
    using Callback = std::function<BatchMatrix(std::span<const int>, const BatchMatrix&, BatchMatrix*)>;
    template <class InputMap, class OutputMap>
    BatchResponseFixture(const InputMap& input, const OutputMap& output, Callback callback, int batch_size = 256)
        : energy::MeshResponse(input, output, batch_size), callback_(std::move(callback)) {
        if (!callback_)
            throw std::invalid_argument("BatchResponseFixture: empty callback");
    }

  protected:
    BatchMatrix eval_batch(std::span<const int> ids, const BatchMatrix& x, BatchMatrix* J) const override {
        return callback_(ids, x, J);
    }

  private:
    Callback callback_;
};

} // namespace litepgo::testing
