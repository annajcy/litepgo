#pragma once

#include "litepgo/core/types.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace litepgo {

// Owns one field's element-major DOF values. Counts come from the vector size
// so they stay consistent across copies and moves.
template <int DofsPerElement>
    requires(DofsPerElement > 0)
class ElementDofData final {
  public:
    // Creates zero data. The element count must be non-negative and the total
    // DOF count fit in int; otherwise throws std::invalid_argument.
    explicit ElementDofData(int num_elements) : values_(Vec::Zero(checked_num_dofs(num_elements))) {}

    // Copies data, inferring the element count. The size must be a multiple of
    // DofsPerElement and fit in int; otherwise throws std::invalid_argument.
    explicit ElementDofData(const ConstVecRef& values) : values_(checked_size(values.size())) {
        values_ = values;
    }

    int num_elements() const noexcept {
        return num_dofs() / DofsPerElement;
    }
    static constexpr int dofs_per_element() noexcept {
        return DofsPerElement;
    }
    int num_dofs() const noexcept {
        return static_cast<int>(values_.size());
    }

    // Views permit editing values without resizing the owned vector. They are
    // valid until this object is assigned, moved from, or destroyed.
    ConstVecRef values() const {
        return values_;
    }
    VecRef values() {
        return values_;
    }

    double operator()(int element, int slot) const {
        return values_[offset(element, slot)];
    }
    double& operator()(int element, int slot) {
        return values_[offset(element, slot)];
    }

  private:
    // element * DofsPerElement + slot. Throws std::out_of_range if either
    // index is invalid.
    int offset(int element, int slot) const {
        if (element < 0 || element >= num_elements()) {
            throw std::out_of_range("ElementDofData::offset: element " + std::to_string(element) + " is out of range");
        }
        if (slot < 0 || slot >= DofsPerElement) {
            throw std::out_of_range("ElementDofData::offset: slot " + std::to_string(slot) + " is out of range");
        }
        return element * DofsPerElement + slot;
    }

    static int checked_num_dofs(int num_elements) {
        if (num_elements < 0 || num_elements > std::numeric_limits<int>::max() / DofsPerElement) {
            throw std::invalid_argument(
                "ElementDofData requires a non-negative element count and a DOF count fitting in int");
        }
        return num_elements * DofsPerElement;
    }

    static Eigen::Index checked_size(Eigen::Index size) {
        if (size > std::numeric_limits<int>::max() || size % DofsPerElement != 0) {
            throw std::invalid_argument(
                "ElementDofData requires a vector size divisible by its element width and fitting in int");
        }
        return size;
    }

    Vec values_;
};

} // namespace litepgo
