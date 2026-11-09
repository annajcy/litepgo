#pragma once
#include "litepgo/dof/response_assembler.hpp"
#include "litepgo/energy/response/response.hpp"
#include <limits>
#include <memory>

namespace litepgo::energy {
// Gather local inputs, transform them, and concatenate outputs in element order.
// The local response is reused sequentially; VJP sums shared/repeated input DOFs.
class ElementwiseResponse final : public Response {
  public:
    template <class Map>
    ElementwiseResponse(const Map& input_map, std::shared_ptr<Response> transform)
        : ElementwiseResponse(make_assembler(input_map, transform), transform) {}

    int num_inputs() const noexcept override {
        return static_cast<int>(assembler_.jacobian_pattern().cols());
    }
    int num_outputs() const noexcept override {
        return static_cast<int>(assembler_.jacobian_pattern().rows());
    }

  private:
    template <class Map>
    static dof::ResponseAssembler make_assembler(const Map& map, const std::shared_ptr<Response>& transform) {
        if (!transform || transform->num_outputs() <= 0 || map.num_elements() < 0 ||
            map.num_elements() > std::numeric_limits<int>::max() / transform->num_outputs())
            throw std::invalid_argument("ElementwiseResponse: invalid transform or element count");
        dof::ResponseAssembler::Indices rows, cols;
        for (int e = 0; e < map.num_elements(); ++e) {
            auto ids = map.dof_indices(e);
            if (ids.size() != static_cast<std::size_t>(transform->num_inputs()))
                throw std::invalid_argument("ElementwiseResponse: incompatible local input dimensions");
            cols.push_back(std::move(ids));
            std::vector<int> out(transform->num_outputs());
            for (int i = 0; i < transform->num_outputs(); ++i)
                out[i] = e * transform->num_outputs() + i;
            rows.push_back(std::move(out));
        }
        return {map.num_elements() * transform->num_outputs(), map.num_dofs(), std::move(rows), std::move(cols)};
    }
    ElementwiseResponse(dof::ResponseAssembler assembler, std::shared_ptr<Response> transform);
    std::shared_ptr<ResponseEvaluation> make_evaluation(const ConstVecRef& x) override;
    dof::ResponseAssembler assembler_;
    std::shared_ptr<Response> transform_;
};
} // namespace litepgo::energy
