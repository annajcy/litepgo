#include "litepgo/fem/local/local_energy.hpp"

#include <stdexcept>

namespace litepgo {

bool LocalEnergy::has_gradient() const noexcept {
    return false;
}
bool LocalEnergy::has_hessian() const noexcept {
    return false;
}

Eigen::VectorXd LocalEnergy::gradient(const RestPointData&) const {
    throw std::logic_error("LocalEnergy does not support gradient evaluation");
}
Eigen::MatrixXd LocalEnergy::hessian(const RestPointData&) const {
    throw std::logic_error("LocalEnergy does not support Hessian evaluation");
}

} // namespace litepgo
