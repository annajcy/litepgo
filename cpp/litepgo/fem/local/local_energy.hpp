#pragma once

#include "litepgo/fem/interpolation/rest_point.hpp"

namespace litepgo {

class LocalEnergy {
  public:
    virtual ~LocalEnergy() = default;
    virtual int num_dofs() const = 0;
    virtual double value(const RestPointData& point) const = 0;

    // Each capability is independent; override both its query and derivative.
    virtual bool has_gradient() const noexcept;
    virtual bool has_hessian() const noexcept;
    virtual Eigen::VectorXd gradient(const RestPointData& point) const;
    virtual Eigen::MatrixXd hessian(const RestPointData& point) const;

    // Adds weighted derivatives into initialized, correctly sized buffers and
    // returns the weighted value. Outputs are independently nullable; they must
    // not be resized or alias energy data. The default uses the methods above.
    virtual double accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                              Eigen::MatrixXd* hessian) const;
};

} // namespace litepgo
