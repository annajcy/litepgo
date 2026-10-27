#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace litepgo {

// Vector-valued field, affine in its active coefficients. Gradient outputs flatten by column.
class Field {
  public:
    virtual ~Field() = default;
    virtual int num_components() const = 0;
    virtual int num_dofs() const = 0;
    virtual Eigen::VectorXd value(const Eigen::Vector3d& xi) const = 0;
    virtual Eigen::MatrixXd dvalue_dcoeffs(const Eigen::Vector3d& xi) const = 0;
    virtual Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi(const Eigen::Vector3d&) const {
        throw std::logic_error("Field does not support reference-coordinate derivatives");
    }
    virtual Eigen::MatrixXd dgrad_xi_dcoeffs(const Eigen::Vector3d&) const {
        throw std::logic_error("Field does not support mixed reference-coordinate and coefficient derivatives");
    }
};

} // namespace litepgo
