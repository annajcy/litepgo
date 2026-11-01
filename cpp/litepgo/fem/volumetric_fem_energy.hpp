#pragma once
#include "litepgo/fem/detail/batch_views.hpp"

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/kinematics/deformation_gradient_kinematics.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"
#include <concepts>
#include <stdexcept>
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
                      auto g = detail::batch_vector_view(gradients, b);
                      auto H = detail::batch_matrix_view(hessians, b, n);
                      const auto index = static_cast<std::size_t>(e);
                      const Eigen::MatrixXd coefficients = transforms_[index].to_coefficients(std::move(u));
                      const Vec flattened = coefficients.reshaped<Eigen::RowMajor>();
                      values[b] = eval_element(index, flattened, g ? &*g : nullptr, H ? &*H : nullptr);
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
                      const Eigen::MatrixXd coefficients = transforms_[index].to_coefficients(u);
                      const Vec flattened = coefficients.reshaped<Eigen::RowMajor>();
                      const Eigen::MatrixXd delta_coefficients = transforms_[index].to_coefficients(du);
                      const Vec delta = delta_coefficients.reshaped<Eigen::RowMajor>();
                      steps[b] = integration_[index].max_feasible_step([&](int q, const auto& point) {
                          const auto input = material_inputs(index, q, point.xi, flattened);
                          const Vec delta_F = kinematics_[index].derivative(q) * delta;
                          if (!delta_F.allFinite())
                              throw std::domain_error("VolumetricFemEnergy: non-finite input increment");
                          const VolumetricMaterialInputs direction{delta_F.reshaped(3, 3), Eigen::Matrix3d::Zero(),
                                                                   Vec::Zero(input.theta.size())};
                          return Material::eval_feasible_step(input, direction);
                      });
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
        kinematics_.reserve(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            const auto index = static_cast<std::size_t>(e);
            if (Fp_[index].num_components() != 9 || theta_[index].num_components() <= 0)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible element fields");
            if (displacement_map.dof_indices(e).size() != 3 * UBasis::num_basis_functions)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible displacement basis");
            integration_.emplace_back(rest_position_fields_[index]);
            transforms_.emplace_back(rest_position_fields_[index]);
            kinematics_.emplace_back(rest_position_fields_[index]);
        }
    }

  private:
    VolumetricMaterialInputs material_inputs(std::size_t e, int q, const Eigen::Vector3d& xi,
                                             const Vec& coefficients) const {
        const auto Fp = Fp_[e].eval_value(xi);
        const auto theta = theta_[e].eval_value(xi);
        if (Fp.size() != 9 || theta.size() == 0 || !Fp.allFinite() || !theta.allFinite())
            throw std::invalid_argument("VolumetricFemEnergy: incompatible finite material inputs required");
        return {kinematics_[e].eval(q, coefficients), Fp.reshaped(3, 3), theta};
    }

    double eval_element(std::size_t e, const Vec& coefficients, StridedVecRef* gradient, StridedMatRef* hessian) const {
        return integration_[e].eval_into(
            static_cast<int>(coefficients.size()),
            [&](int q, const auto& point, StridedVecRef* g, StridedMatRef* H) {
                const auto input = material_inputs(e, q, point.xi, coefficients);
                const double value = point.w_dV * Material::eval_W(input);
                const auto& derivative = kinematics_[e].derivative(q);
                if (g)
                    add_gradient_wrt_dofs(derivative, Material::eval_dW_dF(input), point.w_dV, *g, false);
                if (H)
                    add_hessian_wrt_dofs(derivative, Material::eval_d2W_dF2(input), point.w_dV, *H, false);
                return value;
            },
            gradient, hessian);
    }

    std::vector<DeformationGradientKinematics3D<UBasis, Q>> kinematics_;
    std::vector<FpField> Fp_;
    std::vector<ThetaField> theta_;
    std::vector<XField> rest_position_fields_;
    std::vector<VolumetricElement3D<Q>> integration_;
    std::vector<BasisDofTransform<UBasis>> transforms_;
};

} // namespace litepgo::fem
