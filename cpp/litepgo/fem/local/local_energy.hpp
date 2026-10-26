#pragma once

#include "litepgo/fem/geometry/rest.hpp"

namespace litepgo {

class LocalEnergy {
  public:
    virtual ~LocalEnergy() = default;
    virtual int num_dofs() const = 0;
    // Capabilities are independent and checked before integration changes outputs.
    virtual bool has_gradient() const noexcept {
        return false;
    }
    virtual bool has_hessian() const noexcept {
        return false;
    }

    // Adds weighted derivatives into initialized, correctly sized buffers and
    // returns the weighted value. Outputs are independently nullable; they must
    // not be resized or alias energy data. Both outputs null requests only the value.
    // Implementations must reject non-finite weights and unsupported requests.
    virtual double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                              Eigen::MatrixXd* hessian) const = 0;
};

} // namespace litepgo
