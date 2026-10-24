#pragma once

#include "litepgo/fem/geometry/volume_point_data.hpp"

namespace litepgo {

// Local immutable field snapshot, independent of global DOF numbering.
template <class Value> class Field {
  public:
    virtual ~Field() = default;
    virtual Value value(const VolumePointData& point) const = 0;
};

// At fixed geometry, vec(value) = J * local_dofs + offset.
// vec is column-major. Affineness is a mathematical contract.
template <class Value> class AffineField : public Field<Value> {
  public:
    virtual int num_dofs() const = 0;
    virtual Eigen::MatrixXd jacobian(const VolumePointData& point) const = 0;
};

} // namespace litepgo
