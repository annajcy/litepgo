#pragma once

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include <cmath>
#include <optional>
#include <utility>
#include <vector>

namespace litepgo::fem {

// Triangle mid-edge shell with three displacement DOFs per stencil vertex.
// Owns fixed per-face inputs; the first three stencil rows are the central face.
// Material must remain feasible whenever the geometric step constraints hold
// with fixed thickness and theta (as KoiterStVK does).
template <class Material> class ShellFemEnergy final : public energy::MeshEnergy {
  public:
    // Arrays and stencil rows must match the map's face and vertex order.
    ShellFemEnergy(const dof::StencilDofMap& displacement_map, std::vector<TriangleMidEdgeRestData> rest_data,
                   std::vector<double> thickness, std::vector<Eigen::VectorXd> theta)
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
                      const Eigen::MatrixXd u = inputs.row(b).reshaped<Eigen::RowMajor>(n / 3, 3);
                      std::optional<StridedVecRef> g;
                      std::optional<StridedMatRef> H;
                      if (gradients)
                          g.emplace(gradients->row(b).transpose());
                      if (hessians) {
                          Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> block(
                              hessians->data() + b * n * n, n, n, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(1, n));
                          H.emplace(block);
                      }
                      const auto index = static_cast<std::size_t>(e);
                      const auto& rest = rest_data_[index];
                      const ShellMaterialLocalEnergy2D<Material> local(rest, u, thickness_[index], theta_[index]);
                      if (g)
                          g->setZero();
                      if (H)
                          H->setZero();
                      // This discretization has constant density over the reference triangle.
                      values[b] = local.accumulate_into(Eigen::Vector2d::Constant(1.0 / 3), 0.5 * rest.sqrt_det_a_bar,
                                                        g ? &*g : nullptr, H ? &*H : nullptr);
                  }
                  return values;
              },
              256,
              [this](std::span<const int> ids, const energy::BatchMatrix& inputs, const energy::BatchMatrix& deltas) {
                  Vec steps(ids.size());
                  const Eigen::Index n = inputs.cols();
                  for (Eigen::Index b = 0; b < inputs.rows(); ++b) {
                      const int e = ids[b];
                      const Eigen::MatrixXd u = inputs.row(b).reshaped<Eigen::RowMajor>(n / 3, 3);
                      const Eigen::MatrixXd du = deltas.row(b).reshaped<Eigen::RowMajor>(n / 3, 3);

                      const auto index = static_cast<std::size_t>(e);
                      const auto& rest = rest_data_[index];
                      const ShellMaterialLocalEnergy2D<Material> local(rest, u, thickness_[index], theta_[index]);
                      local.eval_value(Eigen::Vector2d::Constant(1.0 / 3));
                      steps[b] = eval_triangle_mid_edge_feasible_step(rest.X_local + u, du, rest.opposite);
                  }
                  return steps;
              }),
          rest_data_(std::move(rest_data)), thickness_(std::move(thickness)), theta_(std::move(theta)) {
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        if (displacement_map.dofs_per_vertex() != 3 || rest_data_.size() != count || thickness_.size() != count ||
            theta_.size() != count)
            throw std::invalid_argument("ShellFemEnergy: incompatible map or face counts");
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            const auto index = static_cast<std::size_t>(e);
            const auto& rest = rest_data_[index];
            if (displacement_map.vertex_indices(e).size() != static_cast<std::size_t>(rest.X_local.rows()) ||
                !std::isfinite(rest.sqrt_det_a_bar) || rest.sqrt_det_a_bar <= 0)
                throw std::invalid_argument("ShellFemEnergy: incompatible reference stencil or area");
            const ShellMaterialLocalEnergy2D<Material> local(rest, Eigen::MatrixXd::Zero(rest.X_local.rows(), 3),
                                                             thickness_[index], theta_[index]);
            local.eval_value(Eigen::Vector2d::Constant(1.0 / 3));
        }
    }

  private:
    std::vector<TriangleMidEdgeRestData> rest_data_;
    std::vector<double> thickness_;
    std::vector<Eigen::VectorXd> theta_;
};

} // namespace litepgo::fem
