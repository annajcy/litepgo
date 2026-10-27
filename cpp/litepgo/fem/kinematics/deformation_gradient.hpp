#pragma once

#include "litepgo/fem/kinematics/reference_mapping.hpp"
#include <stdexcept>
#include <unsupported/Eigen/KroneckerProduct>
#include <utility>

namespace litepgo {

// Owns field snapshots. Both dependencies are sampled at the same xi; X remains fixed during u differentiation.
template <class UField, class XField> class DeformationGradient final {
    using USample = decltype(std::declval<const UField&>().evaluate(std::declval<const Eigen::Vector3d&>()));

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
            Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + u_sample_.dvalue_dxi * mapping_.dxi_dX();
            if (!F.allFinite())
                throw std::invalid_argument("DeformationGradient::F: non-finite result");
            return F;
        }
        Eigen::MatrixXd dF_du_local() const
            requires(
                requires(const USample& field_sample) { field_sample.basis_sample.dN_dxi; } ||
                requires(const USample& field_sample) { field_sample.dgrad_xi_dcoeffs(); })
        {
            Eigen::MatrixXd derivative;
            if constexpr (requires { u_sample_.basis_sample.dN_dxi; }) {
                const Eigen::MatrixXd dN_dX = u_sample_.basis_sample.dN_dxi * mapping_.dxi_dX();
                if (!dN_dX.allFinite())
                    throw std::invalid_argument("DeformationGradient::dF_du_local: non-finite basis gradients");
                derivative = Eigen::MatrixXd::Zero(9, 3 * dN_dX.rows());
                for (Eigen::Index a = 0; a < dN_dX.rows(); ++a)
                    for (int d = 0; d < 3; ++d)
                        derivative.block<3, 3>(3 * d, 3 * a).diagonal().setConstant(dN_dX(a, d));
            } else {
                const Eigen::MatrixXd dgrad_xi_dcoeffs = u_sample_.dgrad_xi_dcoeffs();
                if (dgrad_xi_dcoeffs.rows() != 9 || dgrad_xi_dcoeffs.cols() <= 0 || !dgrad_xi_dcoeffs.allFinite())
                    throw std::invalid_argument("DeformationGradient::dF_du_local: invalid mixed derivative");
                derivative = Eigen::kroneckerProduct(mapping_.dxi_dX().transpose(), Eigen::Matrix3d::Identity()) *
                             dgrad_xi_dcoeffs;
            }
            if (!derivative.allFinite())
                throw std::invalid_argument("DeformationGradient::dF_du_local: non-finite result");
            return derivative;
        }

      private:
        friend class DeformationGradient;
        Sample(USample u_sample, ReferenceMapping mapping)
            : u_sample_(std::move(u_sample)), mapping_(std::move(mapping)) {
            if (u_sample_.value.size() != 3 || u_sample_.dvalue_dxi.rows() != 3 || u_sample_.dvalue_dxi.cols() != 3 ||
                !u_sample_.value.allFinite() || !u_sample_.dvalue_dxi.allFinite())
                throw std::invalid_argument("DeformationGradient requires three finite displacement components");
        }
        USample u_sample_;
        ReferenceMapping mapping_;
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
        const ReferenceMapping mapping(X_field_.evaluate(xi));
        return Sample(u_field_.evaluate(xi), mapping);
    }

  private:
    UField u_field_;
    XField X_field_;
};

} // namespace litepgo
