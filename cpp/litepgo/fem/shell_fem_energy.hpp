#pragma once

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/mesh_energy.hpp"
#include "litepgo/fem/material/shell/shell_material_local_energy.hpp"
#include <cmath>
#include <utility>
#include <vector>

namespace litepgo {

// Triangle mid-edge shell with three displacement DOFs per stencil vertex.
// Owns fixed per-face inputs; the first three stencil rows are the central face.
// Material must remain feasible whenever the geometric step constraints hold
// with fixed thickness and theta (as KoiterStVK does).
template <class Material> class ShellFemEnergy final : public MeshEnergy {
  public:
    // Arrays and stencil rows must match the map's face and vertex order.
    ShellFemEnergy(const StencilDofMap& displacement_map, std::vector<TriangleMidEdgeRestData> rest_data,
                   std::vector<double> thickness, std::vector<Eigen::VectorXd> theta)
        : MeshEnergy(
              displacement_map,
              [this](int e, const Eigen::MatrixXd& u, Vec* g, Eigen::MatrixXd* H) {
                  const auto index = static_cast<std::size_t>(e);
                  const auto& rest = rest_data_[index];
                  const ShellMaterialLocalEnergy2D<Material> local(rest, u, thickness_[index], theta_[index]);
                  if (g)
                      g->setZero(local.num_dofs());
                  if (H)
                      H->setZero(local.num_dofs(), local.num_dofs());
                  // This discretization has constant density over the reference triangle.
                  return local.accumulate(Eigen::Vector2d::Constant(1.0 / 3), 0.5 * rest.sqrt_det_a_bar, g, H);
              },
              [this](int e, const Eigen::MatrixXd& u, const Eigen::MatrixXd& du) {
                  const auto index = static_cast<std::size_t>(e);
                  const auto& rest = rest_data_[index];
                  const ShellMaterialLocalEnergy2D<Material> local(rest, u, thickness_[index], theta_[index]);
                  local.eval_value(Eigen::Vector2d::Constant(1.0 / 3));
                  return eval_triangle_mid_edge_feasible_step(rest.X_local + u, du, rest.opposite);
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

} // namespace litepgo
