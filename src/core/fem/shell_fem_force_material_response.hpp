#pragma once

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/detail/batch_views.hpp"
#include "litepgo/fem/detail/shell_material_force.hpp"
#include <concepts>
#include <utility>
#include <vector>

namespace litepgo::fem {
// Fixed displacement and thickness; theta is shared globally or constant per face.
// Snapshots all fixed inputs. Outputs follow the displacement stencil's scalar DOF order.
template <class Material> class ShellFemForceMaterialResponse final : public energy::MeshResponse {
  public:
    template <class ParameterMap>
        requires(std::same_as<ParameterMap, dof::ConstantDofMap> || std::same_as<ParameterMap, dof::ElementDofMap>)
    ShellFemForceMaterialResponse(const ParameterMap& parameter_map, const dof::StencilDofMap& displacement_map,
                                  const std::vector<TriangleMidEdgeRestData>& rest_data, const Vec& displacement,
                                  const std::vector<double>& thickness, int batch_size = 256)
        : energy::MeshResponse(parameter_map, displacement_map, batch_size),
          state_(make_state(parameter_map, displacement_map, rest_data, displacement, thickness)) {}

  private:
    struct State {
        std::vector<detail::ShellForceData> data;
        std::vector<double> areas;
    };
    template <class ParameterMap>
    static State make_state(const ParameterMap& parameters, const dof::StencilDofMap& map,
                            const std::vector<TriangleMidEdgeRestData>& rest, const Vec& u,
                            const std::vector<double>& thickness) {
        const auto count = static_cast<std::size_t>(map.num_elements());
        const int width = [&] {
            if constexpr (std::same_as<ParameterMap, dof::ConstantDofMap>)
                return parameters.num_dofs();
            else
                return parameters.dofs_per_element();
        }();
        if (parameters.num_elements() != map.num_elements() || width != Material::num_parameters ||
            map.dofs_per_vertex() != 3 || u.size() != map.num_dofs() || !u.allFinite() || rest.size() != count ||
            thickness.size() != count)
            throw std::invalid_argument("ShellFemForceMaterialResponse: incompatible maps or fixed inputs");
        std::vector<detail::ShellForceData> data;
        std::vector<double> areas;
        data.reserve(count);
        areas.reserve(count);
        for (int e = 0; e < map.num_elements(); ++e) {
            if (map.vertex_indices(e).size() != static_cast<std::size_t>(rest[e].X_local.rows()) ||
                !std::isfinite(rest[e].sqrt_det_a_bar) || rest[e].sqrt_det_a_bar <= 0)
                throw std::invalid_argument("ShellFemForceMaterialResponse: incompatible reference stencil or area");
            data.emplace_back(rest[e], map.gather(e, u), thickness[e]);
            areas.push_back(0.5 * rest[e].sqrt_det_a_bar);
        }
        return {std::move(data), std::move(areas)};
    }
    energy::BatchMatrix eval_batch(std::span<const int> ids, const energy::BatchMatrix& theta,
                                   energy::BatchMatrix* jacobian) const override {
        const auto& data = state_.data;
        const auto& areas = state_.areas;
        const Eigen::Index n = ids.empty() ? 0 : data[ids[0]].dab_du.cols();
        energy::BatchMatrix values = energy::BatchMatrix::Zero(ids.size(), n);
        if (jacobian)
            jacobian->setZero(ids.size() * n, Material::num_parameters);
        for (std::size_t b = 0; b < ids.size(); ++b) {
            const int e = ids[b];
            StridedVecRef value(values.row(b).transpose());
            auto J = detail::batch_matrix_view(jacobian, b, n);
            detail::accumulate_shell_force<Material>(data[e], theta.row(b).transpose(), areas[e], value,
                                                     J ? &*J : nullptr);
        }
        return values;
    }
    State state_;
};
} // namespace litepgo::fem
