#pragma once

#include "litepgo/fem/field.hpp"
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace litepgo {

// Owns a local globally shared value snapshot. DOFs use column-major vec(value).
template <class Value> class ConstantField final : public AffineField<Value> {
    static_assert(std::is_same_v<Value, typename Value::PlainObject> && std::is_same_v<typename Value::Scalar, double>);

  public:
    explicit ConstantField(Value value) : value_(std::move(value)) {
        if (value_.size() == 0 || value_.size() > std::numeric_limits<int>::max() || !value_.allFinite())
            throw std::invalid_argument("ConstantField requires nonempty finite values with int-sized DOFs");
    }
    int num_dofs() const override {
        return static_cast<int>(value_.size());
    }
    Value value(const VolumePointData&) const override {
        return value_;
    }
    Eigen::MatrixXd jacobian(const VolumePointData&) const override {
        return Eigen::MatrixXd::Identity(num_dofs(), num_dofs());
    }

  private:
    Value value_;
};

} // namespace litepgo
