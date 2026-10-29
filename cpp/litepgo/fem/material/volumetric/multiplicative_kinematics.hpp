#pragma once

#include <Eigen/Core>

namespace litepgo::fem {

// A pointwise snapshot of Fe = F Fp^-1 and det_Fp = det(Fp).
class MultiplicativeKinematics final {
  public:
    // Non-finite inputs throw invalid_argument. Non-positive det(Fp), or
    // an unrepresentable inverse/Fe, throw std::domain_error. Fe's domain is
    // the material's responsibility; this class does not require det(F) > 0.
    MultiplicativeKinematics(const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp);

    const Eigen::Matrix3d& elastic_gradient() const {
        return Fe_;
    }
    double plastic_volume_ratio() const {
        return det_Fp_;
    }

    // Convert density per plastic volume to density per original reference
    // volume: W = det_Fp * psi. Derivatives hold Fp fixed and assume dependence on F through Fe.
    // dpsi_dFe retains matrix shape; d2psi_dFe2 uses column-major vec(Fe).
    // Returned gradients retain matrix shape; Hessians use column-major vec(F) or vec(Fp).
    // Non-finite numerical results throw std::domain_error.
    double eval_W(double psi) const;
    Eigen::Matrix3d eval_dW_dF(const Eigen::Matrix3d& dpsi_dFe) const;
    Eigen::Matrix<double, 9, 9> eval_d2W_dF2(const Eigen::Matrix<double, 9, 9>& d2psi_dFe2) const;

    // Hold F and material parameters fixed. psi must depend on Fp only through
    // Fe; all elastic inputs must be evaluated at this snapshot's Fe.
    // Includes derivatives of both det(Fp) and Fp^-1. Same vec convention above.
    Eigen::Matrix3d eval_dW_dFp(double psi, const Eigen::Matrix3d& dpsi_dFe) const;
    Eigen::Matrix<double, 9, 9> eval_d2W_dFp2(double psi, const Eigen::Matrix3d& dpsi_dFe,
                                              const Eigen::Matrix<double, 9, 9>& d2psi_dFe2) const;

  private:
    Eigen::Matrix3d inv_Fp_;
    Eigen::Matrix3d Fe_;
    double det_Fp_;
};

} // namespace litepgo::fem
