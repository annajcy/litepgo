#include "litepgo/fem/local/local_response.hpp"
#include <cmath>
#include <optional>
#include <stdexcept>

namespace litepgo::fem {
template <int Dim> bool LocalResponseT<Dim>::has_jacobian() const noexcept {
    return false;
}
template <int Dim> Eigen::MatrixXd LocalResponseT<Dim>::eval_jacobian(const Coordinate&) const {
    throw std::logic_error("LocalResponseT does not support Jacobian evaluation");
}
template <int Dim>
void LocalResponseT<Dim>::accumulate_into(const Coordinate& xi, double weight, StridedVecRef value,
                                          StridedMatRef* jacobian) const {
    if (!xi.allFinite() || !std::isfinite(weight))
        throw std::invalid_argument("LocalResponseT: invalid coordinates or weight");
    if (jacobian && !has_jacobian())
        throw std::logic_error("LocalResponseT does not support the requested Jacobian");
    const int m = num_outputs(), n = num_inputs();
    if (m < 0 || n < 0 || value.size() != m || (jacobian && (jacobian->rows() != m || jacobian->cols() != n)))
        throw std::invalid_argument("LocalResponseT: invalid output buffer dimensions");
    const Eigen::VectorXd local = eval_value(xi);
    if (local.size() != m)
        throw std::logic_error("LocalResponseT: invalid local value dimensions");
    Eigen::MatrixXd J;
    if (jacobian) {
        J = eval_jacobian(xi);
        if (J.rows() != m || J.cols() != n)
            throw std::logic_error("LocalResponseT: invalid local Jacobian dimensions");
    }
    if (!local.allFinite() || (jacobian && !J.allFinite()))
        throw std::domain_error("LocalResponseT: non-finite local result");
    value += weight * local;
    if (jacobian)
        *jacobian += weight * J;
    if (!value.allFinite() || (jacobian && !jacobian->allFinite()))
        throw std::domain_error("LocalResponseT: non-finite accumulation");
}
template <int ParamDim>
void LocalResponseT<ParamDim>::accumulate(const Coordinate& xi, double weight, Eigen::VectorXd& value,
                                          Eigen::MatrixXd* jacobian) const {
    std::optional<StridedMatRef> J;
    if (jacobian)
        J.emplace(*jacobian);
    accumulate_into(xi, weight, StridedVecRef(value), J ? &*J : nullptr);
}

template class LocalResponseT<2>;
template class LocalResponseT<3>;
} // namespace litepgo::fem
