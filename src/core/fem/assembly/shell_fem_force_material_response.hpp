#pragma once

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/assembly/detail/batch_views.hpp"
#include "litepgo/fem/assembly/detail/shell_material_force.hpp"
#include "litepgo/fem/material/shell/shell_material_batch.hpp"
#include <concepts>
#include <vector>

namespace litepgo::fem {
// Fixed displacement and thickness; theta is shared globally or constant per face.
// Snapshots all fixed inputs. Outputs follow the displacement stencil's scalar DOF order.
// Material is a static pointwise material or a ShellBatchMaterial instance called once per batch.
template <class Material> class ShellFemForceMaterialResponse final : public energy::MeshResponse {
    static constexpr bool batched = ShellBatchMaterial<Material>;

  public:
    template <class ParameterMap>
        requires(std::same_as<ParameterMap, dof::ConstantDofMap> || std::same_as<ParameterMap, dof::ElementDofMap>)
    ShellFemForceMaterialResponse(const ParameterMap& parameter_map, const dof::StencilDofMap& displacement_map,
                                  const std::vector<TriangleMidEdgeRestData>& rest_data, const Vec& displacement,
                                  const std::vector<double>& thickness, int batch_size = 256,
                                  Material material = Material())
        : energy::MeshResponse(parameter_map, displacement_map, batch_size), material_(std::move(material)) {
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        const int width = [&] {
            if constexpr (std::same_as<ParameterMap, dof::ConstantDofMap>)
                return parameter_map.num_dofs();
            else
                return parameter_map.dofs_per_element();
        }();
        if (parameter_map.num_elements() != displacement_map.num_elements() ||
            width != detail::shell_num_parameters(material_) || displacement_map.dofs_per_vertex() != 3 ||
            displacement.size() != displacement_map.num_dofs() || !displacement.allFinite() ||
            rest_data.size() != count || thickness.size() != count)
            throw std::invalid_argument("ShellFemForceMaterialResponse: incompatible maps or fixed inputs");
        faces_.reserve(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            if (displacement_map.vertex_indices(e).size() != static_cast<std::size_t>(rest_data[e].X_local.rows()))
                throw std::invalid_argument("ShellFemForceMaterialResponse: incompatible reference stencil");
            faces_.emplace_back(rest_data[e], displacement_map.gather(e, displacement), thickness[e]);
        }
    }

  private:
    energy::BatchMatrix eval_batch(std::span<const int> ids, const energy::BatchMatrix& theta,
                                   energy::BatchMatrix* jacobian) const override {
        const Eigen::Index n = ids.empty() ? 0 : faces_[ids[0]].dab_du.cols();
        const int k = detail::shell_num_parameters(material_);
        energy::BatchMatrix values(ids.size(), n);
        if (jacobian)
            jacobian->resize(ids.size() * n, k);
        if constexpr (!batched) {
            for (std::size_t b = 0; b < ids.size(); ++b) {
                const int e = ids[b];
                StridedVecRef value(values.row(b).transpose());
                auto J = detail::batch_matrix_view(jacobian, b, n);
                detail::eval_shell_material_force<Material>(faces_[e], theta.row(b).transpose(), value,
                                                            J ? &*J : nullptr);
            }
        } else {
            ShellMaterialBatch batch = detail::make_shell_batch(static_cast<Eigen::Index>(ids.size()), k);
            for (std::size_t b = 0; b < ids.size(); ++b) {
                if (theta.cols() != k || !theta.row(b).allFinite())
                    throw std::invalid_argument("Shell force: invalid parameters, derivative or output dimensions");
                auto inputs = faces_[ids[b]].inputs;
                inputs.theta = theta.row(b).transpose();
                detail::set_shell_point(batch, static_cast<Eigen::Index>(b), inputs);
            }
            Eigen::Matrix<double, 6, Eigen::Dynamic> stress;
            Eigen::MatrixXd mixed;
            ShellMaterialBatchOutputs out{nullptr, &stress};
            if (jacobian)
                out.d2W_dab_dtheta = &mixed;
            material_.eval(batch, out);
            detail::check_shell_batch_outputs(out, batch.size(), k);
            for (std::size_t b = 0; b < ids.size(); ++b) {
                const auto& face = faces_[ids[b]];
                detail::shell_force_value(face, stress.col(b), StridedVecRef(values.row(b).transpose()));
                if (auto J = detail::batch_matrix_view(jacobian, b, n))
                    detail::shell_material_force_jacobian(face, mixed.col(b).reshaped(6, k), k, *J);
            }
        }
        return values;
    }
    Material material_;
    std::vector<detail::ShellForceData> faces_;
};
} // namespace litepgo::fem
