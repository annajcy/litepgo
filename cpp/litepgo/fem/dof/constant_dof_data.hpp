#pragma once
#include "litepgo/core/types.hpp"
#include <stdexcept>
namespace litepgo {
// Owns shared scalar DOFs; interpreting their shape belongs to the local field.
template <int NumDofs>
    requires(NumDofs > 0)
class ConstantDofData final {
  public:
    ConstantDofData() : values_(Eigen::Matrix<double, NumDofs, 1>::Zero()) {}
    explicit ConstantDofData(const ConstVecRef& values) {
        if (values.size() != NumDofs)
            throw std::invalid_argument("ConstantDofData requires exactly NumDofs values");
        values_ = values;
    }
    static constexpr int num_dofs() noexcept {
        return NumDofs;
    }
    ConstVecRef values() const {
        return values_;
    }
    VecRef values() {
        return values_;
    }
    double operator()(int slot) const {
        check_slot(slot);
        return values_[slot];
    }
    double& operator()(int slot) {
        check_slot(slot);
        return values_[slot];
    }

  private:
    static void check_slot(int slot) {
        if (slot < 0 || slot >= NumDofs)
            throw std::out_of_range("ConstantDofData slot is out of range");
    }
    Eigen::Matrix<double, NumDofs, 1> values_;
};
} // namespace litepgo
