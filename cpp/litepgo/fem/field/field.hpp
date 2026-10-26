#pragma once

#include "litepgo/fem/geometry/rest.hpp"

namespace litepgo {

// Local immutable field snapshot, independent of global DOF numbering.
template <class Value> class Field {
  public:
    virtual ~Field() = default;
    virtual Value value(const RestPointData& point) const = 0;
};

// At fixed geometry, vec(value) = dvalue_dx_local * flatten(x_local) + offset.
// Rows index column-major vec(value); columns follow this field's scalar DOF order.
// Affineness is a mathematical contract; x_local's packing is defined by each field.
template <class Value> class AffineField : public Field<Value> {
  public:
    virtual int num_dofs() const = 0;
    virtual Eigen::MatrixXd dvalue_dx_local(const RestPointData& point) const = 0;
};

} // namespace litepgo
