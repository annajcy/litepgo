#pragma once

#include "litepgo/fem/field_value.hpp"

#include <stdexcept>

namespace litepgo {

// A point-independent local snapshot, from an element value or a shared value.
// It does not retain global DOF ownership or numbering. Both the local DOFs and
// returned matrix gradient use column-major vec, including row-major input values.
template <FixedFieldValue ValueType> class DirectLocalField final {
  public:
    using FieldValue = Eigen::Matrix<double, ValueType::RowsAtCompileTime, ValueType::ColsAtCompileTime>;
    static constexpr int num_dofs = FieldValue::SizeAtCompileTime;

    explicit DirectLocalField(const ValueType& value) : value_(value) {
        if (!value_.allFinite())
            throw std::invalid_argument("DirectLocalField requires finite values");
    }
    template <class GeometryPoint> FieldValue value(const GeometryPoint&) const {
        return value_;
    }
    template <class GeometryPoint> static Eigen::Matrix<double, num_dofs, num_dofs> jacobian(const GeometryPoint&) {
        return Eigen::Matrix<double, num_dofs, num_dofs>::Identity();
    }

  private:
    FieldValue value_;
};

} // namespace litepgo
