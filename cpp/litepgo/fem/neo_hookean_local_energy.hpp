#pragma once

#include "litepgo/fem/local_energy.hpp"
#include "litepgo/fem/material_fields.hpp"

namespace litepgo {

// Each class binds one constitutive argument to its active affine field.
class NeoHookeanLocalEnergyF final : public LocalValue, public LocalGradient, public LocalHessian {
  public:
    explicit NeoHookeanLocalEnergyF(MaterialFields fields);
    int num_dofs() const override;
    double value(const VolumePointData& point) const override;
    Eigen::VectorXd gradient(const VolumePointData& point) const override;
    Eigen::MatrixXd hessian(const VolumePointData& point) const override;

  private:
    Eigen::MatrixXd active_jacobian(const VolumePointData& point) const;
    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::Matrix3d>> active_;
};

class NeoHookeanLocalEnergyFp final : public LocalValue, public LocalGradient, public LocalHessian {
  public:
    explicit NeoHookeanLocalEnergyFp(MaterialFields fields);
    int num_dofs() const override;
    double value(const VolumePointData& point) const override;
    Eigen::VectorXd gradient(const VolumePointData& point) const override;
    Eigen::MatrixXd hessian(const VolumePointData& point) const override;

  private:
    Eigen::MatrixXd active_jacobian(const VolumePointData& point) const;
    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::Matrix3d>> active_;
};

class NeoHookeanLocalEnergyTheta final : public LocalValue, public LocalGradient, public LocalHessian {
  public:
    explicit NeoHookeanLocalEnergyTheta(MaterialFields fields);
    int num_dofs() const override;
    double value(const VolumePointData& point) const override;
    Eigen::VectorXd gradient(const VolumePointData& point) const override;
    Eigen::MatrixXd hessian(const VolumePointData& point) const override;

  private:
    Eigen::MatrixXd active_jacobian(const VolumePointData& point) const;
    MaterialFields fields_;
    std::shared_ptr<const AffineField<Eigen::VectorXd>> active_;
};

} // namespace litepgo
