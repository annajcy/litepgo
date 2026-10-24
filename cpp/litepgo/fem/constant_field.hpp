#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/fem/field_value.hpp"

#include <array>

namespace litepgo {

// One spatially constant, globally shared value. It can still be optimized.
// Column-major vec ordering; every element uses the same field-local indices.
// Non-finite storage is allowed; local evaluation validates it.
template <FixedFieldValue Value>
    requires(Value::SizeAtCompileTime > 0)
class ConstantField final {
  public:
    static constexpr int num_dofs = Value::SizeAtCompileTime;
    using Indices = std::array<int, num_dofs>;

    explicit ConstantField(const Value& value) {
        set_value(value);
    }

    Value value() const {
        return Eigen::Map<const PackedValue>(values_.data());
    }
    void set_value(const Value& value) {
        Eigen::Map<PackedValue> block(values_.data());
        block = value;
    }
    Indices dof_indices() const {
        Indices result;
        for (int i = 0; i < num_dofs; ++i)
            result[static_cast<std::size_t>(i)] = i;
        return result;
    }

    // Non-resizable views; the object must outlive them.
    ConstVecRef values() const {
        return values_;
    }
    VecRef values() {
        return values_;
    }

  private:
    using PackedValue = Eigen::Matrix<double, Value::RowsAtCompileTime, Value::ColsAtCompileTime>;
    Eigen::Matrix<double, num_dofs, 1> values_;
};

} // namespace litepgo
