#pragma once

#include "litepgo/fem/material_types.hpp"

namespace litepgo {

// A pointwise snapshot of Fe = F Fp^-1 and Jp = det(Fp).
class MultiplicativeKinematics final {
  public:
    // Non-finite inputs throw invalid_argument. Non-positive det(Fp), or
    // an unrepresentable inverse/Fe, throw InfeasibleState. Fe's domain is
    // the material's responsibility; this class does not require det(F) > 0.
    MultiplicativeKinematics(const Mat3& F, const Mat3& Fp);

    const Mat3& elastic_gradient() const {
        return Fe_;
    }
    double plastic_volume_ratio() const {
        return Jp_;
    }

    // Convert density per plastic volume to density per original reference
    // volume. Derivatives hold Fp fixed and assume dependence on F through Fe.
    // Hessians are exact and use column-major vec(A)[i + 3*j] = A(i,j).
    double reference_energy(double psi) const;
    Mat3 reference_gradient_F(const Mat3& Pe) const;
    Mat9 reference_hessian_F(const Mat9& He) const;

  private:
    Mat3 inverse_Fp_;
    Mat3 Fe_;
    double Jp_;
};

} // namespace litepgo
