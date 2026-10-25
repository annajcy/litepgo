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

Eigen::VectorXd LocalEnergy::gradient(const RestPointData&) const {
    throw std::logic_error("LocalEnergy does not support gradient evaluation");
}
Eigen::MatrixXd LocalEnergy::hessian(const RestPointData&) const {
    throw std::logic_error("LocalEnergy does not support Hessian evaluation");
}

double LocalEnergy::accumulate(const RestPointData& point, double weight, Eigen::VectorXd* gradient,
                               Eigen::MatrixXd* hessian) const {
    if (!std::isfinite(weight))
        throw std::invalid_argument("LocalEnergy::accumulate requires a finite weight");
    if ((gradient && !has_gradient()) || (hessian && !has_hessian()))
        throw std::logic_error("LocalEnergy does not support the requested derivatives");
    const double result = weight * value(point);
    if (gradient) {
        const auto local = this->gradient(point);
        if (local.size() != gradient->size())
            throw std::invalid_argument("LocalEnergy::accumulate: invalid gradient size");
        *gradient += weight * local;
    }
    if (hessian) {
        const auto local = this->hessian(point);
        if (local.rows() != hessian->rows() || local.cols() != hessian->cols())
            throw std::invalid_argument("LocalEnergy::accumulate: invalid Hessian size");
        *hessian += weight * local;
    }
    return result;
}

} // namespace litepgo
