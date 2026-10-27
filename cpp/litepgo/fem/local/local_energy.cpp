#include "litepgo/fem/local/local_energy.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

bool LocalEnergy::has_gradient() const noexcept {
    return false;
}
bool LocalEnergy::has_hessian() const noexcept {
    return false;
}

Eigen::VectorXd LocalEnergy::gradient(const Eigen::Vector3d&) const {
    throw std::logic_error("LocalEnergy does not support gradient evaluation");
}
Eigen::MatrixXd LocalEnergy::hessian(const Eigen::Vector3d&) const {
    throw std::logic_error("LocalEnergy does not support Hessian evaluation");
}

double LocalEnergy::accumulate(const Eigen::Vector3d& xi, double weight, Eigen::VectorXd* gradient,
                               Eigen::MatrixXd* hessian) const {
    if (!xi.allFinite() || !std::isfinite(weight))
        throw std::invalid_argument("LocalEnergy::accumulate requires a finite weight");
    if ((gradient && !has_gradient()) || (hessian && !has_hessian()))
        throw std::logic_error("LocalEnergy does not support the requested derivatives");
    const double result = weight * value(xi);
    if (gradient) {
        const auto local = this->gradient(xi);
        if (local.size() != gradient->size())
            throw std::invalid_argument("LocalEnergy::accumulate: invalid gradient size");
        *gradient += weight * local;
    }
    if (hessian) {
        const auto local = this->hessian(xi);
        if (local.rows() != hessian->rows() || local.cols() != hessian->cols())
            throw std::invalid_argument("LocalEnergy::accumulate: invalid Hessian size");
        *hessian += weight * local;
    }
    return result;
}

} // namespace litepgo
