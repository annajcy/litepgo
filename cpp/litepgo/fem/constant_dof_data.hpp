#pragma once

#include "litepgo/core/types.hpp"
#include "litepgo/fem/field_value.hpp"

#include <array>

namespace litepgo {

// One spatially constant, globally shared value. It can still be optimized.
// Column-major vec ordering; every element uses the same field-local indices.
// Non-finite storage is allowed; local evaluation validates it.
template <FixedFieldValue Value> class ConstantDofData final {
  public:
    static constexpr int dofs_per_value = Value::SizeAtCompileTime;

    explicit ConstantDofData(const Value& value) {
        set_value(value);
    }

    int num_dofs() const noexcept {
        return dofs_per_value;
    }

    Value value() const {
        return Eigen::Map<const Eigen::Matrix<double, Value::RowsAtCompileTime, Value::ColsAtCompileTime>>(
            values_.data());
    }
    void set_value(const Value& value) {
        Eigen::Map<Eigen::Matrix<double, Value::RowsAtCompileTime, Value::ColsAtCompileTime>> block(values_.data());
        block = value;
    }
    std::array<int, dofs_per_value> dof_indices() const {
        std::array<int, dofs_per_value> result;
        for (int i = 0; i < dofs_per_value; ++i)
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
    Eigen::Matrix<double, dofs_per_value, 1> values_;
};

} // namespace litepgo
