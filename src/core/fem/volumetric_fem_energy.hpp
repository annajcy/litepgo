#pragma once
#include "litepgo/fem/detail/batch_views.hpp"

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"
#include "litepgo/fem/response/sampling_response.hpp"
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace litepgo::fem {

// Geometry and fixed material responses are sampled at construction; rebuild after changing them.
template <class Material, Basis3D UBasis, Quadrature3D Q> class VolumetricFemEnergy final : public energy::MeshEnergy {
  public:
    // One spatial response (input xi in R3) per element, in displacement-map order.
    VolumetricFemEnergy(const dof::VertexDofMap& displacement_map,
                        const std::vector<std::shared_ptr<energy::Response>>& rest_positions,
                        const std::vector<std::shared_ptr<energy::Response>>& Fp,
                        const std::vector<std::shared_ptr<energy::Response>>& theta)
        : energy::MeshEnergy(displacement_map, 256) {
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        if (displacement_map.dofs_per_vertex() != BasisDofTransform<UBasis>::dofs_per_vertex ||
            rest_positions.size() != count || Fp.size() != count || theta.size() != count)
            throw std::invalid_argument("VolumetricFemEnergy: incompatible map or response counts");
        integration_.reserve(count);
        transforms_.reserve(count);
        deformation_jacobians_.reserve(count);
        Fp_.resize(count);
        theta_.resize(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            if (!rest_positions[e] || !Fp[e] || !theta[e] || Fp[e]->num_inputs() != 3 || theta[e]->num_inputs() != 3 ||
                Fp[e]->num_outputs() != 9 || theta[e]->num_outputs() != Material::num_parameters)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible spatial responses");
            if (displacement_map.dof_indices(e).size() != 3 * UBasis::num_basis_functions)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible displacement basis");
            integration_.emplace_back(*rest_positions[e]);
            transforms_.emplace_back(*rest_positions[e]);
            deformation_jacobians_.emplace_back();
            auto sampling = deformation_gradient_sampling_response<UBasis>(*rest_positions[e], Q::points());
            const Eigen::MatrixXd A = sampling.eval_jacobian(Vec::Zero(3 * UBasis::num_basis_functions));
            for (int q = 0; q < Q::num_points; ++q)
                deformation_jacobians_.back()[q] = A.middleRows(9 * q, 9);
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Vector3d xi = Q::points().row(q).transpose();
                const Vec plastic = Fp[e]->eval_value(xi);
                Fp_[e][q] = plastic.reshaped(3, 3);
                theta_[e][q] = theta[e]->eval_value(xi);
            }
        }
    }

  private:
    Vec eval_batch(std::span<const int> ids, const energy::BatchMatrix& inputs, energy::BatchMatrix* gradients,
                   energy::BatchMatrix* hessians) const override {
        const Eigen::Index count = inputs.rows(), n = inputs.cols();
        Vec values(count);
        if (gradients)
            gradients->resize(count, n);
        if (hessians)
            hessians->resize(count * n, n);
        for (Eigen::Index b = 0; b < count; ++b) {
            const int e = ids[b];
            Eigen::MatrixXd u = inputs.row(b).reshaped<Eigen::RowMajor>(n / BasisDofTransform<UBasis>::dofs_per_vertex,
                                                                        BasisDofTransform<UBasis>::dofs_per_vertex);
            auto g = detail::batch_vector_view(gradients, b);
            auto H = detail::batch_matrix_view(hessians, b, n);
            const auto index = static_cast<std::size_t>(e);
            const Eigen::MatrixXd coefficients = transforms_[index].to_coefficients(std::move(u));
            const Vec flattened = coefficients.reshaped<Eigen::RowMajor>();
            values[b] = eval_element(index, flattened, g ? &*g : nullptr, H ? &*H : nullptr);
            transforms_[index].pullback_into(g ? &*g : nullptr, H ? &*H : nullptr);
        }
        return values;
    }
    Vec eval_feasible_step_batch(std::span<const int> ids, const energy::BatchMatrix& inputs,
                                 const energy::BatchMatrix& deltas) const override {
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
            steps[b] = integration_[index].max_feasible_step([&](int q, const auto&) {
                const auto input = material_inputs(index, q, flattened);
                const Vec delta_F = deformation_jacobians_[index][q] * delta;
                if (!delta_F.allFinite())
                    throw std::domain_error("VolumetricFemEnergy: non-finite input increment");
                const VolumetricMaterialInputs direction{delta_F.reshaped(3, 3), Eigen::Matrix3d::Zero(),
                                                         Vec::Zero(input.theta.size())};
                return Material::eval_feasible_step(input, direction);
            });
        }
        return steps;
    }
    VolumetricMaterialInputs material_inputs(std::size_t e, int q, const Vec& coefficients) const {
        const Vec increment = deformation_jacobians_[e][q] * coefficients;
        const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + increment.reshaped(3, 3);
        if (!F.allFinite())
            throw std::domain_error("VolumetricFemEnergy: non-finite deformation gradient");
        return {F, Fp_[e][q], theta_[e][q]};
    }

    double eval_element(std::size_t e, const Vec& coefficients, StridedVecRef* gradient, StridedMatRef* hessian) const {
        return integration_[e].eval_into(
            static_cast<int>(coefficients.size()),
            [&](int q, const auto& point, StridedVecRef* g, StridedMatRef* H) {
                const auto input = material_inputs(e, q, coefficients);
                const double value = point.w_dV * Material::eval_W(input);
                const auto& derivative = deformation_jacobians_[e][q];
                if (g)
                    add_gradient_wrt_dofs(derivative, Material::eval_dW_dF(input), point.w_dV, *g, false);
                if (H)
                    add_hessian_wrt_dofs(derivative, Material::eval_d2W_dF2(input), point.w_dV, *H, false);
                return value;
            },
            gradient, hessian);
    }

    std::vector<std::array<Eigen::MatrixXd, Q::num_points>> deformation_jacobians_;
    std::vector<std::array<Eigen::Matrix3d, Q::num_points>> Fp_;
    std::vector<std::array<Vec, Q::num_points>> theta_;
    std::vector<VolumetricElement3D<Q>> integration_;
    std::vector<BasisDofTransform<UBasis>> transforms_;
};

} // namespace litepgo::fem
