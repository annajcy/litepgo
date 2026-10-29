#include "litepgo/fem/local/local_energy.hpp"

#include <cmath>
#include <stdexcept>

namespace litepgo::fem {

template <int ParamDim> bool LocalEnergyT<ParamDim>::has_gradient() const noexcept {
    return false;
}
template <int ParamDim> bool LocalEnergyT<ParamDim>::has_hessian() const noexcept {
    return false;
}

template <int ParamDim> Eigen::VectorXd LocalEnergyT<ParamDim>::eval_gradient(const Coordinate&) const {
    throw std::logic_error("LocalEnergyT does not support gradient evaluation");
}
template <int ParamDim> Eigen::MatrixXd LocalEnergyT<ParamDim>::eval_hessian(const Coordinate&) const {
    throw std::logic_error("LocalEnergyT does not support Hessian evaluation");
}

template <int ParamDim>
double LocalEnergyT<ParamDim>::accumulate(const Coordinate& xi, double weight, Eigen::VectorXd* gradient,
                                          Eigen::MatrixXd* hessian) const {
    if (!xi.allFinite() || !std::isfinite(weight))
        throw std::invalid_argument("LocalEnergyT::accumulate requires a finite weight");
    if ((gradient && !has_gradient()) || (hessian && !has_hessian()))
        throw std::logic_error("LocalEnergyT does not support the requested derivatives");
    const double result = weight * eval_value(xi);
    if (!std::isfinite(result))
        throw std::domain_error("LocalEnergyT::accumulate: non-finite weighted density");
    if (gradient) {
        const auto local = this->eval_gradient(xi);
        if (local.size() != gradient->size())
            throw std::invalid_argument("LocalEnergyT::accumulate: invalid gradient size");
        *gradient += weight * local;
        if (!gradient->allFinite())
            throw std::domain_error("LocalEnergyT::accumulate: non-finite gradient accumulation");
    }
    if (hessian) {
        const auto local = this->eval_hessian(xi);
        if (local.rows() != hessian->rows() || local.cols() != hessian->cols())
            throw std::invalid_argument("LocalEnergyT::accumulate: invalid Hessian size");
        *hessian += weight * local;
        if (!hessian->allFinite())
            throw std::domain_error("LocalEnergyT::accumulate: non-finite Hessian accumulation");
    }
    return result;
}

template class LocalEnergyT<2>;
template class LocalEnergyT<3>;

} // namespace litepgo::fem
