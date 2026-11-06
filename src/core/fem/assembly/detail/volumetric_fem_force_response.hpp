#pragma once
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/assembly/detail/batch_views.hpp"
#include "litepgo/fem/assembly/basis_dof_transform.hpp"
#include "litepgo/fem/integration/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"
#include "litepgo/fem/response/sampling_response.hpp"
#include <array>
#include <concepts>
#include <memory>
#include <vector>

namespace litepgo::fem {
namespace detail {
enum class ForceInput { Material, Plastic };
// Fixed displacement and geometry; the other material input is stored per element.
template <class Material, Basis3D UBasis, Quadrature3D Q, ForceInput Input>
class VolumetricFemForceResponse final : public energy::MeshResponse {
    static constexpr int output_size = 3 * UBasis::num_basis_functions;
    static constexpr int input_size = Input == ForceInput::Plastic ? 9 : Material::num_parameters;
    static constexpr int fixed_size = Input == ForceInput::Plastic ? Material::num_parameters : 9;

  public:
    template <class ParameterMap>
        requires(std::same_as<ParameterMap, dof::ConstantDofMap> || std::same_as<ParameterMap, dof::ElementDofMap>)
    VolumetricFemForceResponse(const ParameterMap& parameter_map, const dof::VertexDofMap& displacement_map,
                               std::vector<std::shared_ptr<energy::Response>> rest_position_fields, Vec displacement,
                               std::vector<std::shared_ptr<energy::Response>> fixed_fields, int batch_size = 256)
        : energy::MeshResponse(parameter_map, displacement_map, batch_size) {
        if (parameter_map.num_elements() != displacement_map.num_elements() || batch_size <= 0)
            throw std::invalid_argument("VolumetricFemForceResponse: incompatible element count or batch size");
        const int width = [&] {
            if constexpr (std::same_as<ParameterMap, dof::ConstantDofMap>)
                return parameter_map.num_dofs();
            else
                return parameter_map.dofs_per_element();
        }();
        if (width != input_size)
            throw std::invalid_argument("VolumetricFemForceResponse: invalid parameter count");
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        if (displacement_map.dofs_per_vertex() != BasisDofTransform<UBasis>::dofs_per_vertex ||
            displacement.size() != displacement_map.num_dofs() || !displacement.allFinite() ||
            rest_position_fields.size() != count || fixed_fields.size() != count)
            throw std::invalid_argument("VolumetricFemForceResponse: incompatible map, displacement or responses");
        fixed_.resize(count);
        deformation_.resize(count);
        deformation_jacobians_.reserve(count);
        integration_.reserve(count);
        transforms_.reserve(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            if (!rest_position_fields[e] || !fixed_fields[e] || displacement_map.dof_indices(e).size() != output_size ||
                fixed_fields[e]->num_inputs() != 3 || fixed_fields[e]->num_outputs() != fixed_size)
                throw std::invalid_argument("VolumetricFemForceResponse: incompatible spatial response or basis");
            transforms_.emplace_back(*rest_position_fields[e]);
            deformation_jacobians_.emplace_back();
            auto sampling = deformation_gradient_sampling_response<UBasis>(*rest_position_fields[e], Q::points());
            const Eigen::MatrixXd A = sampling.eval_jacobian(Vec::Zero(output_size));
            for (int q = 0; q < Q::num_points; ++q)
                deformation_jacobians_.back()[q] = A.middleRows(9 * q, 9);
            integration_.emplace_back(*rest_position_fields[e]);
            const Eigen::MatrixXd coefficients =
                transforms_.back().to_coefficients(displacement_map.gather(e, displacement));
            const Vec flattened = coefficients.reshaped<Eigen::RowMajor>();
            const Vec values = sampling.eval_value(flattened);
            for (int q = 0; q < Q::num_points; ++q) {
                deformation_[e][q] = values.segment<9>(9 * q).reshaped(3, 3);
                fixed_[e][q] = fixed_fields[e]->eval_value(Q::points().row(q).transpose());
            }
        }
    }

  private:
    energy::BatchMatrix eval_batch(std::span<const int> ids, const energy::BatchMatrix& coefficients,
                                   energy::BatchMatrix* jacobian) const override {
        energy::BatchMatrix values(ids.size(), output_size);
        if (jacobian)
            jacobian->resize(output_size * ids.size(), input_size);
        for (std::size_t b = 0; b < ids.size(); ++b) {
            const int e = ids[b];
            StridedVecRef value(values.row(b).transpose());
            auto J = batch_matrix_view(jacobian, b, output_size);
            value.setZero();
            if (J)
                J->setZero();
            const Vec active = coefficients.row(b).transpose();
            for (int q = 0; q < Q::num_points; ++q) {
                VolumetricMaterialInputs inputs;
                inputs.F = deformation_[e][q];
                if constexpr (Input == ForceInput::Plastic) {
                    inputs.Fp = active.reshaped(3, 3);
                    inputs.theta = fixed_[e][q];
                } else {
                    inputs.Fp = fixed_[e][q].reshaped(3, 3);
                    inputs.theta = active;
                }
                const auto& B = deformation_jacobians_[e][q];
                const double weight = integration_[e].point(q).w_dV;
                const auto stress = Material::eval_dW_dF(inputs);
                if (stress.rows() != 3 || stress.cols() != 3)
                    throw std::logic_error("VolumetricFemForceResponse: invalid material stress dimensions");
                if (!stress.allFinite())
                    throw std::domain_error("VolumetricFemForceResponse: non-finite material stress");
                value.noalias() += weight * B.transpose() * stress.reshaped();
                if (J) {
                    const Eigen::MatrixXd mixed = [&] {
                        if constexpr (Input == ForceInput::Plastic)
                            return Material::eval_d2W_dF_dFp(inputs);
                        else
                            return Material::eval_d2W_dF_dtheta(inputs);
                    }();
                    if (mixed.rows() != 9 || mixed.cols() != input_size)
                        throw std::logic_error("VolumetricFemForceResponse: invalid material mixed derivative");
                    J->noalias() += weight * B.transpose() * mixed;
                }
            }
            transforms_[e].pullback_response_into(value, J ? &*J : nullptr);
        }
        return values;
    }

    std::vector<std::array<Vec, Q::num_points>> fixed_;
    std::vector<std::array<Eigen::Matrix3d, Q::num_points>> deformation_;
    std::vector<std::array<Eigen::MatrixXd, Q::num_points>> deformation_jacobians_;
    std::vector<VolumetricElement3D<Q>> integration_;
    std::vector<BasisDofTransform<UBasis>> transforms_;
};
} // namespace detail
} // namespace litepgo::fem
