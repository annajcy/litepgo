#pragma once
#include <memory>

#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/assembly/detail/batch_views.hpp"
#include "litepgo/fem/assembly/detail/shell_fp_force.hpp"
#include "litepgo/fem/material/shell/shell_material_batch.hpp"
#include <concepts>
#include <vector>

namespace litepgo::fem {
// Fixed original area weights, displacement, thickness and theta; Fp is shared or constant per face.
// Inputs are column-major vec(Fp). Plastic natural areas never replace the original integration weights.
// Snapshots all fixed inputs. Outputs follow the displacement stencil's scalar DOF order.
// Material is evaluated once per batch of faces.
class ShellFemForceFpResponse final : public energy::MeshResponse {

  public:
    template <class ParameterMap>
        requires(std::same_as<ParameterMap, dof::ConstantDofMap> || std::same_as<ParameterMap, dof::ElementDofMap>)
    ShellFemForceFpResponse(std::shared_ptr<const ShellBatchMaterial> material, const ParameterMap& parameter_map,
                            const dof::StencilDofMap& displacement_map,
                            const std::vector<TriangleMidEdgeRestData>& rest_data, const Vec& displacement,
                            const std::vector<double>& thickness, const std::vector<Vec>& theta, int batch_size = 256)
        : energy::MeshResponse(parameter_map, displacement_map, batch_size), material_(std::move(material)) {
        if (!material_ || material_->num_parameters() < 0)
            throw std::invalid_argument("ShellFemForceFpResponse: invalid material");
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
            if (theta[e].size() != material_->num_parameters())
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

        const int k = material_->num_parameters();
        std::vector<detail::ShellFpNatural> natural;
        natural.reserve(ids.size());
        ShellMaterialBatch batch = detail::make_shell_batch(static_cast<Eigen::Index>(ids.size()), k);
        for (std::size_t b = 0; b < ids.size(); ++b) {
            const auto& face = faces_[ids[b]];
            const Vec packed = plastic.row(b).transpose();
            auto J = detail::batch_matrix_view(jacobian, b, n);
            detail::check_shell_fp_input(face, packed, k, n, J ? &*J : nullptr);
            natural.push_back(detail::shell_fp_natural(face, packed, jacobian != nullptr));
            detail::set_shell_point(batch, static_cast<Eigen::Index>(b), natural.back().inputs);
        }
        Eigen::Matrix<double, 6, Eigen::Dynamic> stress;
        Eigen::Matrix<double, 36, Eigen::Dynamic> mixed;
        ShellMaterialBatchOutputs out{nullptr, &stress};
        if (jacobian)
            out.d2W_dab_dabarbbar = &mixed;
        material_->eval(batch, out);
        detail::check_shell_batch_outputs(out, batch.size(), k);
        for (std::size_t b = 0; b < ids.size(); ++b) {
            const auto& face = faces_[ids[b]];
            detail::shell_force_value(face.current, stress.col(b), StridedVecRef(values.row(b).transpose()));
            if (auto J = detail::batch_matrix_view(jacobian, b, n))
                detail::shell_fp_force_jacobian(face, natural[b], mixed.col(b).reshaped(6, 6), *J);
        }

        return values;
    }
    std::shared_ptr<const ShellBatchMaterial> material_;
    std::vector<detail::ShellFpForceData> faces_;
};
} // namespace litepgo::fem
