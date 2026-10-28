#include "litepgo/fem/local/local_energy.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo {

template <int ParamDim> bool LocalEnergyT<ParamDim>::has_gradient() const noexcept {
    return false;
}
template <int ParamDim> bool LocalEnergyT<ParamDim>::has_hessian() const noexcept {
    return false;
}

template <int ParamDim> Eigen::VectorXd LocalEnergyT<ParamDim>::gradient(const Coordinate&) const {
    throw std::logic_error("LocalEnergyT does not support gradient evaluation");
}
template <int ParamDim> Eigen::MatrixXd LocalEnergyT<ParamDim>::hessian(const Coordinate&) const {
    throw std::logic_error("LocalEnergyT does not support Hessian evaluation");
}

template <int ParamDim>
double LocalEnergyT<ParamDim>::accumulate(const Coordinate& xi, double weight, Eigen::VectorXd* gradient,
                                          Eigen::MatrixXd* hessian) const {
    if (!xi.allFinite() || !std::isfinite(weight))
        throw std::invalid_argument("LocalEnergyT::accumulate requires a finite weight");
    if ((gradient && !has_gradient()) || (hessian && !has_hessian()))
        throw std::logic_error("LocalEnergyT does not support the requested derivatives");
    const double result = weight * value(xi);
    if (gradient) {
        const auto local = this->gradient(xi);
        if (local.size() != gradient->size())
            throw std::invalid_argument("LocalEnergyT::accumulate: invalid gradient size");
        *gradient += weight * local;
    }
    if (hessian) {
        const auto local = this->hessian(xi);
        if (local.rows() != hessian->rows() || local.cols() != hessian->cols())
            throw std::invalid_argument("LocalEnergyT::accumulate: invalid Hessian size");
        *hessian += weight * local;
    }
    return result;
}

template class LocalEnergyT<2>;
template class LocalEnergyT<3>;

} // namespace litepgo
