#pragma once
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/detail/batch_views.hpp"
#include "litepgo/fem/kinematics/basis_dof_transform.hpp"
#include "litepgo/fem/kinematics/deformation_gradient_kinematics.hpp"
#include "litepgo/fem/kinematics/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"
#include <concepts>
#include <memory>
#include <utility>

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
        : energy::MeshResponse(parameter_map, displacement_map, batch_size),
          state_(make_state(parameter_map, displacement_map, std::move(rest_position_fields), std::move(displacement),
                            std::move(fixed_fields), batch_size)) {}

  private:
    struct State {
        std::vector<std::array<Vec, Q::num_points>> fixed;
        std::vector<std::array<Eigen::Matrix3d, Q::num_points>> deformation;
        std::vector<DeformationGradientKinematics3D<UBasis, Q>> kinematics;
        std::vector<VolumetricElement3D<Q>> integration;
        std::vector<BasisDofTransform<UBasis>> transforms;
        State(const dof::VertexDofMap& map, const std::vector<std::shared_ptr<energy::Response>>& geometry,
              const Vec& u, const std::vector<std::shared_ptr<energy::Response>>& fixed_responses) {
            const auto count = static_cast<std::size_t>(map.num_elements());
            if (map.dofs_per_vertex() != BasisDofTransform<UBasis>::dofs_per_vertex || u.size() != map.num_dofs() ||
                !u.allFinite() || geometry.size() != count || fixed_responses.size() != count)
                throw std::invalid_argument("VolumetricFemForceResponse: incompatible map, displacement or responses");
            fixed.resize(count);
            deformation.resize(count);
            kinematics.reserve(count);
            integration.reserve(count);
            transforms.reserve(count);
            for (int e = 0; e < map.num_elements(); ++e) {
                if (!geometry[e] || !fixed_responses[e] || map.dof_indices(e).size() != output_size ||
                    fixed_responses[e]->num_inputs() != 3 || fixed_responses[e]->num_outputs() != fixed_size)
                    throw std::invalid_argument("VolumetricFemForceResponse: incompatible spatial response or basis");
                transforms.emplace_back(*geometry[e]);
                kinematics.emplace_back(*geometry[e]);
                integration.emplace_back(*geometry[e]);
                const Eigen::MatrixXd coefficients = transforms.back().to_coefficients(map.gather(e, u));
                const Vec flattened = coefficients.reshaped<Eigen::RowMajor>();
                for (int q = 0; q < Q::num_points; ++q) {
                    deformation[e][q] = kinematics.back().eval(q, flattened);
                    fixed[e][q] = fixed_responses[e]->eval_value(Q::points().row(q).transpose());
                }
            }
        }
        energy::BatchMatrix evaluate(std::span<const int> ids, const energy::BatchMatrix& coefficients,
                                     energy::BatchMatrix* jacobian) const {
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
                    inputs.F = deformation[e][q];
                    if constexpr (Input == ForceInput::Plastic) {
                        inputs.Fp = active.reshaped(3, 3);
                        inputs.theta = fixed[e][q];
                    } else {
                        inputs.Fp = fixed[e][q].reshaped(3, 3);
                        inputs.theta = active;
                    }
                    const auto& B = kinematics[e].derivative(q);
                    const double weight = integration[e].point(q).w_dV;
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
                transforms[e].pullback_response_into(value, J ? &*J : nullptr);
            }
            return values;
        }
    };
    template <class ParameterMap>
    static State make_state(const ParameterMap& parameters, const dof::VertexDofMap& map,
                            std::vector<std::shared_ptr<energy::Response>> X, Vec u,
                            std::vector<std::shared_ptr<energy::Response>> fixed_fields, int batch_size) {
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
        return State(map, std::move(X), u, std::move(fixed_fields));
    }
    energy::BatchMatrix eval_batch(std::span<const int> ids, const energy::BatchMatrix& x,
                                   energy::BatchMatrix* J) const override {
        return state_.evaluate(ids, x, J);
    }
    State state_;
};
} // namespace detail
} // namespace litepgo::fem
