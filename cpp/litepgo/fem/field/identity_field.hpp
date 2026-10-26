#pragma once

#include "litepgo/fem/field/field.hpp"
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace litepgo {

// Owns a position-independent value snapshot: vec(value) equals flattened x_local.
// Matrix DOFs use column-major ordering, regardless of Value storage order.
template <class Value> class IdentityField final : public AffineField<Value> {
    static_assert(std::is_same_v<Value, typename Value::PlainObject> && std::is_same_v<typename Value::Scalar, double>);

  public:
    explicit IdentityField(Value value) : value_(std::move(value)) {
        if (value_.size() == 0 || value_.size() > std::numeric_limits<int>::max() || !value_.allFinite())
            throw std::invalid_argument("IdentityField requires nonempty finite values with int-sized DOFs");
    }
    int num_dofs() const override {
        return static_cast<int>(value_.size());
    }
    Value value(const RestPointData&) const override {
        return value_;
    }
    Eigen::MatrixXd dvalue_dx_local(const RestPointData&) const override {
        return Eigen::MatrixXd::Identity(num_dofs(), num_dofs());
    }

  private:
    Value value_;
};

} // namespace litepgo
