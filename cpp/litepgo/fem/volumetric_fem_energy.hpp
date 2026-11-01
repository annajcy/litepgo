#pragma once

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include <concepts>
#include <optional>
#include <utility>
#include <vector>

namespace litepgo::fem {

// BasisDofTransform defines the physical displacement DOFs. Fixed fields and rest geometry
// are owned; fields that borrow data require that data to outlive the energy.
// Field evaluations must support concurrent const calls.
template <class Material, Basis3D UBasis, Quadrature3D Q, std::derived_from<Field3D> XField,
          std::derived_from<Field3D> FpField, std::derived_from<Field3D> ThetaField>
class VolumetricFemEnergy final : public energy::MeshEnergy {
  public:
    // Each field vector has one entry per element in map order, using the same
    // reference coordinates as UBasis. Reference geometry need not use UBasis.
    VolumetricFemEnergy(const dof::VertexDofMap& displacement_map, std::vector<XField> rest_position_fields,
                        std::vector<FpField> Fp, std::vector<ThetaField> theta)
        : energy::MeshEnergy(
              displacement_map,
              [this](std::span<const int> ids, const energy::BatchMatrix& inputs, energy::BatchMatrix* gradients,
                     energy::BatchMatrix* hessians) {
                  const Eigen::Index count = inputs.rows(), n = inputs.cols();
                  Vec values(count);
                  if (gradients)
                      gradients->resize(count, n);
                  if (hessians)
                      hessians->resize(count * n, n);
                  for (Eigen::Index b = 0; b < count; ++b) {
                      const int e = ids[b];
                      Eigen::MatrixXd u = inputs.row(b).reshaped<Eigen::RowMajor>(
                          n / BasisDofTransform<UBasis>::dofs_per_vertex, BasisDofTransform<UBasis>::dofs_per_vertex);
                      std::optional<StridedVecRef> g;
                      std::optional<StridedMatRef> H;
                      if (gradients)
                          g.emplace(gradients->row(b).transpose());
                      if (hessians) {
                          Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> block(
                              hessians->data() + b * n * n, n, n, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
                          H.emplace(block);
                      }
                      const auto index = static_cast<std::size_t>(e);
                      const CachedDeformationGradient F(transforms_[index].to_coefficients(std::move(u)),
                                                        derivatives_[index]);
                      const VolumetricMaterialLocalEnergyF3D<Material> local(F, Fp_[index], theta_[index]);
                      values[b] = integration_[index].eval_into(local, g ? &*g : nullptr, H ? &*H : nullptr);
                      transforms_[index].pullback_into(g ? &*g : nullptr, H ? &*H : nullptr);
                  }
                  return values;
              },
              256,
              [this](std::span<const int> ids, const energy::BatchMatrix& inputs, const energy::BatchMatrix& deltas) {
                  Vec steps(ids.size());
                  const Eigen::Index n = inputs.cols();
                  for (Eigen::Index b = 0; b < inputs.rows(); ++b) {
                      const int e = ids[b];
                      const Eigen::MatrixXd u = inputs.row(b).reshaped<Eigen::RowMajor>(
                          n / BasisDofTransform<UBasis>::dofs_per_vertex, BasisDofTransform<UBasis>::dofs_per_vertex);
                      const Eigen::MatrixXd du = deltas.row(b).reshaped<Eigen::RowMajor>(
                          n / BasisDofTransform<UBasis>::dofs_per_vertex, BasisDofTransform<UBasis>::dofs_per_vertex);

                      const auto index = static_cast<std::size_t>(e);
                      const CachedDeformationGradient F(transforms_[index].to_coefficients(u), derivatives_[index]);
                      const VolumetricMaterialLocalEnergyF3D<Material> local(F, Fp_[index], theta_[index]);
                      const auto& delta_coefficients = transforms_[index].to_coefficients(du);
                      const Vec delta = delta_coefficients.template reshaped<Eigen::RowMajor>();
                      steps[b] = integration_[index].max_feasible_step(local, delta);
                  }
                  return steps;
              }),
          Fp_(std::move(Fp)), theta_(std::move(theta)), rest_position_fields_(std::move(rest_position_fields)) {
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        if (displacement_map.dofs_per_vertex() != BasisDofTransform<UBasis>::dofs_per_vertex ||
            rest_position_fields_.size() != count || Fp_.size() != count || theta_.size() != count)
            throw std::invalid_argument("VolumetricFemEnergy: incompatible map or field counts");
        integration_.reserve(count);
        transforms_.reserve(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            const auto index = static_cast<std::size_t>(e);
            if (Fp_[index].num_components() != 9 || theta_[index].num_components() <= 0)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible element fields");
            if (displacement_map.dof_indices(e).size() != 3 * UBasis::num_basis_functions)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible displacement basis");
            integration_.emplace_back(rest_position_fields_[index]);
            transforms_.emplace_back(rest_position_fields_[index]);
            const InterpolatedField3D<UBasis> zero(Eigen::MatrixXd::Zero(UBasis::num_basis_functions, 3));
            const DeformationGradientField3D F(zero, rest_position_fields_[index]);
            derivatives_.emplace_back();
            for (int q = 0; q < Q::num_points; ++q)
                derivatives_.back()[q] = F.eval_dvalue_dcoeffs(Q::points().row(q).transpose());
        }
    }

  private:
    using Derivatives = std::array<Eigen::MatrixXd, Q::num_points>;
    // Fixed geometry and basis make F affine in the current coefficients.
    class CachedDeformationGradient final : public Field3D {
      public:
        CachedDeformationGradient(const Eigen::MatrixXd& u, const Derivatives& derivatives)
            : coefficients_(u.template reshaped<Eigen::RowMajor>()), derivatives_(derivatives) {}
        int num_components() const override {
            return 9;
        }
        int num_dofs() const override {
            return static_cast<int>(coefficients_.size());
        }
        Eigen::VectorXd eval_value(const Eigen::Vector3d& xi) const override {
            // The cached Jacobian also supplies physical gradients without reevaluating X.
            Eigen::VectorXd value = derivatives_[index(xi)] * coefficients_;
            value += Eigen::Matrix3d::Identity().reshaped();
            return value;
        }
        Eigen::MatrixXd eval_dvalue_dcoeffs(const Eigen::Vector3d& xi) const override {
            return derivatives_[index(xi)];
        }

      private:
        static int index(const Eigen::Vector3d& xi) {
            for (int q = 0; q < Q::num_points; ++q)
                if ((Q::points().row(q).transpose().array() == xi.array()).all())
                    return q;
            throw std::invalid_argument("Cached deformation gradient requires a quadrature point");
        }
        Eigen::VectorXd coefficients_;
        const Derivatives& derivatives_;
    };
    std::vector<Derivatives> derivatives_;
    std::vector<FpField> Fp_;
    std::vector<ThetaField> theta_;
    std::vector<XField> rest_position_fields_;
    std::vector<VolumetricElement3D<Q>> integration_;
    std::vector<BasisDofTransform<UBasis>> transforms_;
};

} // namespace litepgo::fem
