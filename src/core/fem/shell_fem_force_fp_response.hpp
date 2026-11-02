#pragma once

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/detail/batch_views.hpp"
#include "litepgo/fem/material/shell/shell_fp_force_local_response.hpp"
#include <concepts>
#include <utility>
#include <vector>

namespace litepgo::fem {
// Fixed original area weights, displacement, thickness and theta; Fp is shared or constant per face.
// Inputs are column-major vec(Fp). Plastic natural areas never replace the original integration weights.
// Snapshots all fixed inputs. Outputs follow the displacement stencil's scalar DOF order.
template <class Material> class ShellFemForceFpResponse final : public energy::MeshResponse {
  public:
    template <class ParameterMap>
        requires(std::same_as<ParameterMap, dof::ConstantDofMap> || std::same_as<ParameterMap, dof::ElementDofMap>)
    ShellFemForceFpResponse(const ParameterMap& parameter_map, const dof::StencilDofMap& displacement_map,
                            const std::vector<TriangleMidEdgeRestData>& rest_data, const Vec& displacement,
                            const std::vector<double>& thickness, const std::vector<Vec>& theta, int batch_size = 256)
        : energy::MeshResponse(
              parameter_map, displacement_map,
              make_callback(parameter_map, displacement_map, rest_data, displacement, thickness, theta), batch_size) {}

  private:
    template <class ParameterMap>
    static energy::MeshResponse::Callback make_callback(const ParameterMap& parameters, const dof::StencilDofMap& map,
                                                        const std::vector<TriangleMidEdgeRestData>& rest, const Vec& u,
                                                        const std::vector<double>& thickness,
                                                        const std::vector<Vec>& theta) {
        const auto count = static_cast<std::size_t>(map.num_elements());
        const int width = [&] {
            if constexpr (std::same_as<ParameterMap, dof::ConstantDofMap>)
                return parameters.num_dofs();
            else
                return parameters.dofs_per_element();
        }();
        if (parameters.num_elements() != map.num_elements() || width != 9 || map.dofs_per_vertex() != 3 ||
            u.size() != map.num_dofs() || !u.allFinite() || rest.size() != count || thickness.size() != count ||
            theta.size() != count)
            throw std::invalid_argument("ShellFemForceFpResponse: incompatible maps or fixed inputs");
        std::vector<detail::ShellFpForceData> data;
        std::vector<double> areas;
        data.reserve(count);
        areas.reserve(count);
        for (int e = 0; e < map.num_elements(); ++e) {
            if (map.vertex_indices(e).size() != static_cast<std::size_t>(rest[e].X_local.rows()) ||
                !std::isfinite(rest[e].sqrt_det_a_bar) || rest[e].sqrt_det_a_bar <= 0)
                throw std::invalid_argument("ShellFemForceFpResponse: incompatible reference stencil or area");
            if (theta[e].size() != Material::num_parameters)
                throw std::invalid_argument("ShellFemForceFpResponse: invalid material parameter count");
            data.emplace_back(rest[e], map.gather(e, u), thickness[e], theta[e]);
            areas.push_back(0.5 * rest[e].sqrt_det_a_bar);
        }
        return [data = std::move(data), areas = std::move(areas)](
                   std::span<const int> ids, const energy::BatchMatrix& plastic, energy::BatchMatrix* jacobian) {
            const Eigen::Index n = ids.empty() ? 0 : data[ids[0]].current.dab_du.cols();
            energy::BatchMatrix values = energy::BatchMatrix::Zero(ids.size(), n);
            if (jacobian)
                jacobian->setZero(ids.size() * n, 9);
            for (std::size_t b = 0; b < ids.size(); ++b) {
                const int e = ids[b];
                StridedVecRef value(values.row(b).transpose());
                auto J = detail::batch_matrix_view(jacobian, b, n);
                detail::accumulate_shell_fp_force<Material>(data[e], plastic.row(b).transpose(), nullptr, areas[e],
                                                            value, J ? &*J : nullptr);
            }
            return values;
        };
    }
};
} // namespace litepgo::fem
