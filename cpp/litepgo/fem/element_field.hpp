#pragma once

#include "litepgo/fem/element_dof_data.hpp"

#include <stdexcept>

namespace litepgo {

// Owns a single element snapshot, without global DOF numbering.
// Local DOFs and vec(value) use column-major order, regardless of input storage order.
template <FixedFieldValue Value> class ElementField final {
  public:
    using FieldValue = Eigen::Matrix<double, Value::RowsAtCompileTime, Value::ColsAtCompileTime>;
    static constexpr int num_dofs = FieldValue::SizeAtCompileTime;

    explicit ElementField(const Value& value) : value_(value) {
        if (!value_.allFinite())
            throw std::invalid_argument("ElementField requires finite values");
    }

    explicit ElementField(const ElementDofData<Value>& data, int element) : ElementField(data.value(element)) {}

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
