#pragma once

#include <Eigen/Core>

namespace litepgo {

// A pointwise snapshot of Fe = F Fp^-1 and Jp = det(Fp).
class MultiplicativeKinematics final {
  public:
    // Non-finite inputs throw invalid_argument. Non-positive det(Fp), or
    // an unrepresentable inverse/Fe, throw InfeasibleState. Fe's domain is
    // the material's responsibility; this class does not require det(F) > 0.
    MultiplicativeKinematics(const Eigen::Matrix3d& F, const Eigen::Matrix3d& Fp);

    const Eigen::Matrix3d& elastic_gradient() const {
        return Fe_;
    }
    double plastic_volume_ratio() const {
        return Jp_;
    }

    // Convert density per plastic volume to density per original reference
    // volume. Derivatives hold Fp fixed and assume dependence on F through Fe.
    // Hessians are exact and use column-major vec(A)[i + 3*j] = A(i,j).
    double reference_energy(double psi) const;
    Eigen::Matrix3d reference_gradient_F(const Eigen::Matrix3d& Pe) const;
    Eigen::Matrix<double, 9, 9> reference_hessian_F(const Eigen::Matrix<double, 9, 9>& He) const;

    // Hold F and material parameters fixed. psi must depend on Fp only through
    // Fe; all elastic inputs must be evaluated at this snapshot's Fe.
    // Includes derivatives of both det(Fp) and Fp^-1. Same vec convention above.
    Eigen::Matrix3d reference_gradient_Fp(double psi, const Eigen::Matrix3d& Pe) const;
    Eigen::Matrix<double, 9, 9> reference_hessian_Fp(double psi, const Eigen::Matrix3d& Pe,
                                                     const Eigen::Matrix<double, 9, 9>& He) const;

  private:
    Eigen::Matrix3d inverse_Fp_;
    Eigen::Matrix3d Fe_;
    double Jp_;
};

} // namespace litepgo
