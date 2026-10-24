#include "litepgo/fem/neo_hookean_local_energy.hpp"
#include "litepgo/fem/detail/dof_derivatives.hpp"
#include "litepgo/fem/neo_hookean.hpp"

#include <stdexcept>
#include <utility>

namespace litepgo {

NeoHookeanLocalEnergyF::NeoHookeanLocalEnergyF(MaterialFields fields) : fields_(std::move(fields)) {
    fields_.validate();
    active_ = std::dynamic_pointer_cast<const AffineField<Eigen::Matrix3d>>(fields_.F);
    if (!active_ || active_->num_dofs() <= 0)
        throw std::invalid_argument("NeoHookeanLocalEnergyF requires an affine F field with positive DOFs");
}

int NeoHookeanLocalEnergyF::num_dofs() const {
    return active_->num_dofs();
}

double NeoHookeanLocalEnergyF::value(const VolumePointData& point) const {
    return NeoHookean{}.value(fields_.evaluate(point));
}

Eigen::VectorXd NeoHookeanLocalEnergyF::gradient(const VolumePointData& point) const {
    return detail::gradient_wrt_dofs(active_jacobian(point), NeoHookean{}.gradient_F(fields_.evaluate(point)));
}

Eigen::MatrixXd NeoHookeanLocalEnergyF::hessian(const VolumePointData& point) const {
    return detail::hessian_wrt_dofs(active_jacobian(point), NeoHookean{}.hessian_F(fields_.evaluate(point)));
}

Eigen::MatrixXd NeoHookeanLocalEnergyF::active_jacobian(const VolumePointData& point) const {
    auto J = active_->jacobian(point);
    if (J.rows() != 9 || J.cols() != num_dofs() || !J.allFinite())
        throw std::invalid_argument("NeoHookeanLocalEnergyF: invalid active Jacobian");
    return J;
}

NeoHookeanLocalEnergyFp::NeoHookeanLocalEnergyFp(MaterialFields fields) : fields_(std::move(fields)) {
    fields_.validate();
    active_ = std::dynamic_pointer_cast<const AffineField<Eigen::Matrix3d>>(fields_.Fp);
    if (!active_ || active_->num_dofs() <= 0)
        throw std::invalid_argument("NeoHookeanLocalEnergyFp requires an affine Fp field with positive DOFs");
}

int NeoHookeanLocalEnergyFp::num_dofs() const {
    return active_->num_dofs();
}

double NeoHookeanLocalEnergyFp::value(const VolumePointData& point) const {
    return NeoHookean{}.value(fields_.evaluate(point));
}

Eigen::VectorXd NeoHookeanLocalEnergyFp::gradient(const VolumePointData& point) const {
    return detail::gradient_wrt_dofs(active_jacobian(point), NeoHookean{}.gradient_Fp(fields_.evaluate(point)));
}

Eigen::MatrixXd NeoHookeanLocalEnergyFp::hessian(const VolumePointData& point) const {
    return detail::hessian_wrt_dofs(active_jacobian(point), NeoHookean{}.hessian_Fp(fields_.evaluate(point)));
}

Eigen::MatrixXd NeoHookeanLocalEnergyFp::active_jacobian(const VolumePointData& point) const {
    auto J = active_->jacobian(point);
    if (J.rows() != 9 || J.cols() != num_dofs() || !J.allFinite())
        throw std::invalid_argument("NeoHookeanLocalEnergyFp: invalid active Jacobian");
    return J;
}

NeoHookeanLocalEnergyTheta::NeoHookeanLocalEnergyTheta(MaterialFields fields) : fields_(std::move(fields)) {
    fields_.validate();
    active_ = std::dynamic_pointer_cast<const AffineField<Eigen::VectorXd>>(fields_.theta);
    if (!active_ || active_->num_dofs() <= 0)
        throw std::invalid_argument("NeoHookeanLocalEnergyTheta requires an affine theta field with positive DOFs");
}

int NeoHookeanLocalEnergyTheta::num_dofs() const {
    return active_->num_dofs();
}

double NeoHookeanLocalEnergyTheta::value(const VolumePointData& point) const {
    return NeoHookean{}.value(fields_.evaluate(point));
}

Eigen::VectorXd NeoHookeanLocalEnergyTheta::gradient(const VolumePointData& point) const {
    return detail::gradient_wrt_dofs(active_jacobian(point), NeoHookean{}.gradient_theta(fields_.evaluate(point)));
}

Eigen::MatrixXd NeoHookeanLocalEnergyTheta::hessian(const VolumePointData& point) const {
    return detail::hessian_wrt_dofs(active_jacobian(point), NeoHookean{}.hessian_theta(fields_.evaluate(point)));
}

Eigen::MatrixXd NeoHookeanLocalEnergyTheta::active_jacobian(const VolumePointData& point) const {
    auto J = active_->jacobian(point);
    if (J.rows() != 2 || J.cols() != num_dofs() || !J.allFinite())
        throw std::invalid_argument("NeoHookeanLocalEnergyTheta: invalid active Jacobian");
    return J;
}

} // namespace litepgo
