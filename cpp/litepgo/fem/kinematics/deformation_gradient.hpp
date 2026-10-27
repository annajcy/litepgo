#pragma once

#include "litepgo/fem/kinematics/reference_mapping.hpp"
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

// Owns field snapshots. Both dependencies are sampled at the same xi; X remains fixed during u differentiation.
template <class UField, class XField> class DeformationGradient final {
  public:
    class Sample final {
      public:
        const Eigen::Vector3d& xi() const {
            return mapping_.xi();
        }
        const ReferenceMapping& reference_mapping() const {
            return mapping_;
        }
        double det_dX_dxi() const {
            return mapping_.det_dX_dxi();
        }
        Eigen::Matrix3d F() const {
            Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + dvalue_dxi_ * mapping_.dxi_dX();
            if (!F.allFinite())
                throw std::invalid_argument("DeformationGradient::F: non-finite result");
            return F;
        }
        Eigen::MatrixXd dF_du_local() const
            requires requires(const UField& field) { field.dgrad_xi_dcoeffs(Eigen::Vector3d{}); }
        {
            // shortcut: materializes the mixed Jacobian; specialize if field composition becomes a bottleneck.
            const Eigen::MatrixXd dgrad_xi_dcoeffs = u_field_.dgrad_xi_dcoeffs(mapping_.xi());
            if (dgrad_xi_dcoeffs.rows() != 9 || dgrad_xi_dcoeffs.cols() <= 0 || !dgrad_xi_dcoeffs.allFinite())
                throw std::invalid_argument("DeformationGradient::dF_du_local: invalid mixed derivative");
            Eigen::MatrixXd derivative =
                Eigen::kroneckerProduct(mapping_.dxi_dX().transpose(), Eigen::Matrix3d::Identity()) * dgrad_xi_dcoeffs;
            if (!derivative.allFinite())
                throw std::invalid_argument("DeformationGradient::dF_du_local: non-finite result");
            return derivative;
        }

      private:
        friend class DeformationGradient;
        Sample(UField u_field, ReferenceMapping mapping)
            : u_field_(std::move(u_field)), mapping_(std::move(mapping)),
              dvalue_dxi_(u_field_.dvalue_dxi(mapping_.xi())) {
            const auto u = u_field_.value(mapping_.xi());
            if (u.size() != 3 || dvalue_dxi_.rows() != 3 || dvalue_dxi_.cols() != 3 || !u.allFinite() ||
                !dvalue_dxi_.allFinite())
                throw std::invalid_argument("DeformationGradient requires three finite displacement components");
        }
        UField u_field_;
        ReferenceMapping mapping_;
        Eigen::Matrix<double, Eigen::Dynamic, 3> dvalue_dxi_;
    };
    DeformationGradient(UField u_field, XField X_field) : u_field_(std::move(u_field)), X_field_(std::move(X_field)) {
        if (u_field_.num_components() != 3 || X_field_.num_components() != 3)
            throw std::invalid_argument("DeformationGradient requires three displacement and coordinate components");
    }
    int num_dofs() const {
        if constexpr (requires { u_field_.num_dofs(); })
            return u_field_.num_dofs();
        else
            return 0;
    }
    Sample evaluate(const Eigen::Vector3d& xi) const {
        const ReferenceMapping mapping(X_field_, xi);
        return Sample(u_field_, mapping);
    }

  private:
    UField u_field_;
    XField X_field_;
};

} // namespace litepgo
