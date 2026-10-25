#pragma once

#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/material/material_inputs.hpp"

#include <memory>

namespace litepgo {

// Shared immutable local snapshots, independent of their concrete representation.
struct MaterialFields {
    std::shared_ptr<const Field<Eigen::Matrix3d>> F;
    std::shared_ptr<const Field<Eigen::Matrix3d>> Fp;
    std::shared_ptr<const Field<Eigen::VectorXd>> theta;

    void validate() const;
    MaterialInputs evaluate(const RestPointData& point) const;
};

} // namespace litepgo
