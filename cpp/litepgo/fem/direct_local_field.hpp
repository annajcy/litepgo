#pragma once

#include "litepgo/fem/field_value.hpp"

#include <stdexcept>

namespace litepgo {

// A point-independent local snapshot, from an element value or a shared value.
// It does not retain global DOF ownership or numbering. Both the local DOFs and
// returned matrix gradient use column-major vec, including row-major input values.
template <FixedFieldValue ValueType> class DirectLocalField final {
  public:
    using Value = Eigen::Matrix<double, ValueType::RowsAtCompileTime, ValueType::ColsAtCompileTime>;
    static constexpr int num_dofs = Value::SizeAtCompileTime;
    using Jacobian = Eigen::Matrix<double, num_dofs, num_dofs>;

    explicit DirectLocalField(const ValueType& value) : value_(value) {
        if (!value_.allFinite())
            throw std::invalid_argument("DirectLocalField requires finite values");
    }
    template <class ReferencePoint> Value value(const ReferencePoint&) const {
        return value_;
    }
    template <class ReferencePoint> static Jacobian jacobian(const ReferencePoint&) {
        return Jacobian::Identity();
    }

  private:
    Value value_;
};

} // namespace litepgo
