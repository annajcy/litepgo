#pragma once
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/volumetric_force_local_response.hpp"
#include "litepgo/response/mesh_response.hpp"
#include <concepts>
#include <memory>
#include <utility>

namespace litepgo::fem {
namespace detail {
// Fixed displacement and geometry; the other material input is stored per element.
template <class Material, Basis3D UBasis, Quadrature3D Q, std::derived_from<Field3D> XField,
          std::derived_from<Field3D> FixedField, ForceInput Input>
class VolumetricFemForceResponse final : public response::FixedPatternResponse {
    static constexpr int output_size = 3 * UBasis::num_basis_functions;
    static constexpr int input_size = Input == ForceInput::Plastic ? 9 : Material::num_parameters;
    static constexpr int fixed_size = Input == ForceInput::Plastic ? Material::num_parameters : 9;

  public:
    template <class ParameterMap>
        requires(std::same_as<ParameterMap, dof::ConstantDofMap> || std::same_as<ParameterMap, dof::ElementDofMap>)
    VolumetricFemForceResponse(const ParameterMap& parameter_map, const dof::VertexDofMap& displacement_map,
                               std::vector<XField> rest_position_fields, Vec displacement,
                               std::vector<FixedField> fixed_fields, int batch_size = 256)
        : VolumetricFemForceResponse(make_response(parameter_map, displacement_map, std::move(rest_position_fields),
                                                   std::move(displacement), std::move(fixed_fields), batch_size)) {}

  private:
    struct State {
        std::vector<XField> geometry;
        std::vector<FixedField> fixed;
        std::vector<InterpolatedField3D<UBasis>> displacement;
        std::vector<VolumetricElement3D<Q>> integration;
        std::vector<BasisDofTransform<UBasis>> transforms;
        State(const dof::VertexDofMap& map, std::vector<XField> X, const Vec& u, std::vector<FixedField> fixed_fields)
            : geometry(std::move(X)), fixed(std::move(fixed_fields)) {
            const auto count = static_cast<std::size_t>(map.num_elements());
            if (map.dofs_per_vertex() != BasisDofTransform<UBasis>::dofs_per_vertex || u.size() != map.num_dofs() ||
                !u.allFinite() || geometry.size() != count || fixed.size() != count)
                throw std::invalid_argument("VolumetricFemForceResponse: incompatible map, displacement or fields");
            displacement.reserve(count);
            integration.reserve(count);
            transforms.reserve(count);
            for (int e = 0; e < map.num_elements(); ++e) {
                if (map.dof_indices(e).size() != output_size || fixed[e].num_components() != fixed_size)
                    throw std::invalid_argument(
                        "VolumetricFemForceResponse: incompatible basis DOFs or fixed field components");
                transforms.emplace_back(geometry[e]);
                displacement.emplace_back(transforms.back().to_coefficients(map.gather(e, u)));
                integration.emplace_back(geometry[e]);
            }
        }
        response::BatchMatrix evaluate(std::span<const int> ids, const response::BatchMatrix& coefficients,
                                       response::BatchMatrix* jacobian) const {
            response::BatchMatrix values(ids.size(), output_size);
            if (jacobian)
                jacobian->resize(output_size * ids.size(), input_size);
            for (std::size_t b = 0; b < ids.size(); ++b) {
                const int e = ids[b];
                const DeformationGradientField3D F(displacement[e], geometry[e]);
                const IdentityField3D active(coefficients.row(b).transpose());
                const auto local = [&] {
                    if constexpr (Input == ForceInput::Plastic)
                        return VolumetricForceLocalResponse3D<Material, Input>(F, active, fixed[e]);
                    else
                        return VolumetricForceLocalResponse3D<Material, Input>(F, fixed[e], active);
                }();
                StridedVecRef value(values.row(b).transpose());
                if (jacobian) {
                    // Expose the row-major batch block through the strided output API.
                    Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> block(
                        jacobian->data() + output_size * b * input_size, output_size, input_size,
                        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, input_size));
                    StridedMatRef J(block);
                    integration[e].eval_into(local, value, &J);
                    transforms[e].pullback_response_into(value, &J);
                } else {
                    integration[e].eval_into(local, value);
                    transforms[e].pullback_response_into(value, nullptr);
                }
            }
            return values;
        }
    };
    template <class ParameterMap>
    static std::unique_ptr<response::MeshResponse>
    make_response(const ParameterMap& parameters, const dof::VertexDofMap& map, std::vector<XField> X, Vec u,
                  std::vector<FixedField> fixed_fields, int batch_size) {
        if (parameters.num_elements() != map.num_elements() || batch_size <= 0)
            throw std::invalid_argument("VolumetricFemForceResponse: incompatible element count or batch size");
        const int width = [&] {
            if constexpr (std::same_as<ParameterMap, dof::ConstantDofMap>)
                return parameters.num_dofs();
            else
                return parameters.dofs_per_element();
        }();
        if (width != input_size)
            throw std::invalid_argument("VolumetricFemForceResponse: invalid parameter count");
        auto state = std::make_shared<State>(map, std::move(X), u, std::move(fixed_fields));
        return std::make_unique<response::MeshResponse>(
            parameters, map,
            [state](std::span<const int> ids, const response::BatchMatrix& coefficients, response::BatchMatrix* J) {
                return state->evaluate(ids, coefficients, J);
            },
            batch_size);
    }
    explicit VolumetricFemForceResponse(std::unique_ptr<response::MeshResponse> assembled)
        : FixedPatternResponse(assembled->jacobian_pattern()), assembled_(std::move(assembled)) {}
    Vec eval_values(const ConstVecRef& coefficients, SpMat* J) const override {
        return assembled_->eval(coefficients, J);
    }
    std::unique_ptr<response::MeshResponse> assembled_;
};
} // namespace detail
} // namespace litepgo::fem
