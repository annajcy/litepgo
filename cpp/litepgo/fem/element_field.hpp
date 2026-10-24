#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/fem/field_value.hpp"

#include <array>
#include <limits>
#include <span>
#include <stdexcept>

namespace litepgo {

// Owns one independent value per element. Each block is column-major vec(value),
// regardless of Value's storage order. std::array<int, dofs_per_element> are local to this field's vector.
// Storage permits non-finite values, like VertexDofData; local evaluation validates them.
template <FixedFieldValue Value> class ElementField final {
  public:
    static constexpr int dofs_per_element = Value::SizeAtCompileTime;

    explicit ElementField(int num_elements) : values_(Vec::Zero(checked_num_dofs(num_elements))) {}

    // Copies the values; never retains the span.
    explicit ElementField(std::span<const Value> values) : ElementField(checked_count(values.size())) {
        for (int e = 0; e < num_elements(); ++e)
            set_value(e, values[static_cast<std::size_t>(e)]);
    }

    int num_elements() const noexcept {
        return num_dofs() / dofs_per_element;
    }
    int num_dofs() const noexcept {
        return static_cast<int>(values_.size());
    }

    Value value(int element) const {
        return Eigen::Map<const Eigen::Matrix<double, Value::RowsAtCompileTime, Value::ColsAtCompileTime>>(
            values_.data() + offset(element));
    }

    void set_value(int element, const Value& value) {
        Eigen::Map<Eigen::Matrix<double, Value::RowsAtCompileTime, Value::ColsAtCompileTime>> block(values_.data() +
                                                                                                    offset(element));
        block = value;
    }

    std::array<int, dofs_per_element> dof_indices(int element) const {
        const int first = offset(element);
        std::array<int, dofs_per_element> result;
        for (int i = 0; i < dofs_per_element; ++i)
            result[static_cast<std::size_t>(i)] = first + i;
        return result;
    }

    // Non-resizable views, valid until assignment, move, or destruction.
    ConstVecRef values() const {
        return values_;
    }
    VecRef values() {
        return values_;
    }

  private:
    static int checked_num_dofs(int count) {
        if (count < 0 || count > std::numeric_limits<int>::max() / dofs_per_element)
            throw std::invalid_argument(
                "ElementField requires a non-negative element count and DOF count fitting in int");
        return count * dofs_per_element;
    }
    static int checked_count(std::size_t count) {
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max() / dofs_per_element))
            throw std::invalid_argument("ElementField DOF count must fit in int");
        return static_cast<int>(count);
    }
    int offset(int element) const {
        if (element < 0 || element >= num_elements())
            throw std::out_of_range("ElementField element is out of range");
        return element * dofs_per_element;
    }

    Vec values_;
};

} // namespace litepgo
