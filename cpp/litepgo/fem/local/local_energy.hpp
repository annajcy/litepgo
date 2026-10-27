#pragma once

#include <Eigen/Core>

namespace litepgo {

// Pointwise density W and derivatives with respect to one declared, affine coefficient block.
class LocalEnergy {
  public:
    virtual ~LocalEnergy() = default;
    virtual int num_dofs() const = 0;
    virtual double value(const Eigen::Vector3d& xi) const = 0;
    virtual double det_dX_dxi(const Eigen::Vector3d& xi) const = 0;
    virtual bool has_gradient() const noexcept;
    virtual bool has_hessian() const noexcept;
    virtual Eigen::VectorXd gradient(const Eigen::Vector3d& xi) const;
    virtual Eigen::MatrixXd hessian(const Eigen::Vector3d& xi) const;

    // quadrature_weight is in parameter space; applies det(dX/dxi) exactly once.
    // Adds requested derivatives to initialized, non-aliasing buffers without resizing them.
    virtual double accumulate(const Eigen::Vector3d& xi, double quadrature_weight, Eigen::VectorXd* gradient,
                              Eigen::MatrixXd* hessian) const;
};

} // namespace litepgo
