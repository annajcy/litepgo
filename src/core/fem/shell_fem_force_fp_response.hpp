#pragma once

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/detail/batch_views.hpp"
#include "litepgo/fem/detail/shell_fp_force.hpp"
#include <concepts>
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
        : energy::MeshResponse(parameter_map, displacement_map, batch_size) {
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        const int width = [&] {
            if constexpr (std::same_as<ParameterMap, dof::ConstantDofMap>)
                return parameter_map.num_dofs();
            else
                return parameter_map.dofs_per_element();
        }();
        if (parameter_map.num_elements() != displacement_map.num_elements() || width != 9 ||
            displacement_map.dofs_per_vertex() != 3 || displacement.size() != displacement_map.num_dofs() ||
            !displacement.allFinite() || rest_data.size() != count || thickness.size() != count ||
            theta.size() != count)
            throw std::invalid_argument("ShellFemForceFpResponse: incompatible maps or fixed inputs");
        faces_.reserve(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            if (displacement_map.vertex_indices(e).size() != static_cast<std::size_t>(rest_data[e].X_local.rows()))
                throw std::invalid_argument("ShellFemForceFpResponse: incompatible reference stencil");
            if (theta[e].size() != Material::num_parameters)
                throw std::invalid_argument("ShellFemForceFpResponse: invalid material parameter count");
            faces_.emplace_back(rest_data[e], displacement_map.gather(e, displacement), thickness[e], theta[e]);
        }
    }

  private:
    energy::BatchMatrix eval_batch(std::span<const int> ids, const energy::BatchMatrix& plastic,
                                   energy::BatchMatrix* jacobian) const override {
        const Eigen::Index n = ids.empty() ? 0 : faces_[ids[0]].current.dab_du.cols();
        energy::BatchMatrix values(ids.size(), n);
        if (jacobian)
            jacobian->resize(ids.size() * n, 9);
        for (std::size_t b = 0; b < ids.size(); ++b) {
            const int e = ids[b];
            StridedVecRef value(values.row(b).transpose());
            auto J = detail::batch_matrix_view(jacobian, b, n);
            detail::eval_shell_fp_force<Material>(faces_[e], plastic.row(b).transpose(), value, J ? &*J : nullptr);
        }
        return values;
    }
    std::vector<detail::ShellFpForceData> faces_;
};
} // namespace litepgo::fem
