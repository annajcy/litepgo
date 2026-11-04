#pragma once
#include "litepgo/dof/response_assembler.hpp"
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include <array>
#include <limits>
#include <memory>

namespace litepgo::energy {
namespace binding {
template <class T> struct ResponseOwnerGC;
}
// Gather local inputs, transform them, and concatenate outputs in element order.
// The local response is reused sequentially; VJP sums shared/repeated input DOFs.
class ElementwiseResponse final : public FixedPatternResponse {
  public:
    template <class Map>
    ElementwiseResponse(const Map& input_map, std::shared_ptr<Response> transform)
        : ElementwiseResponse(make_assembler(input_map, transform), transform) {}

  private:
    template <class T> friend struct binding::ResponseOwnerGC;
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
    Response& transform() const;
    Vec gather(int e, const ConstVecRef& x) const;
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override;
    Vec eval_jvp_values(const ConstVecRef& x, const ConstVecRef& v) const override;
    Vec eval_vjp_values(const ConstVecRef& x, const ConstVecRef& w) const override;
    dof::ResponseAssembler assembler_;
    std::array<std::shared_ptr<Response>, 1> children_;
};
} // namespace litepgo::energy
