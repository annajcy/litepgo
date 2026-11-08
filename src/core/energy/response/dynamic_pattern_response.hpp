#pragma once
#include "litepgo/energy/response/detail/sparse_response.hpp"
namespace litepgo::energy {
class DynamicPatternResponse : public detail::SparseResponse {
  public:
    int num_inputs() const noexcept final {
        return num_inputs_;
    }
    int num_outputs() const noexcept final {
        return num_outputs_;
    }

  protected:
    virtual SpMat jacobian_pattern_at(const ConstVecRef& x) const = 0;
    virtual SpMat hessian_pattern_at(const ConstVecRef&) const {
        throw std::logic_error("Response does not implement Hessian pattern");
    }
    DynamicPatternResponse(int num_inputs, int num_outputs) : num_inputs_(num_inputs), num_outputs_(num_outputs) {
        if (num_inputs < 0 || num_outputs < 0)
            throw std::invalid_argument("Response: negative dimensions");
    }

  private:
    SpMat make_jacobian_pattern(const ConstVecRef& x) const final {
        return checked_pattern(jacobian_pattern_at(x), num_outputs(), num_inputs());
    }
    SpMat make_hessian_pattern(const ConstVecRef& x) const final {
        return checked_pattern(hessian_pattern_at(x), num_inputs(), num_inputs(), true);
    }
    int num_inputs_, num_outputs_;
};
} // namespace litepgo::energy
