#pragma once
#include "litepgo/fem/assembly/detail/batch_views.hpp"
#include "litepgo/fem/assembly/detail/shell_plastic_geometry.hpp"
#include <memory>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/fem/assembly/detail/shell_element.hpp"
#include "litepgo/fem/material/shell/shell_material_batch.hpp"
#include <cmath>
#include <utility>
#include <vector>

namespace litepgo::fem {

// Triangle mid-edge shell with three displacement DOFs per stencil vertex.
// Owns fixed per-face inputs; the first three stencil rows are the central face.
// Material must remain feasible whenever the geometric step constraints hold
// with fixed thickness and theta. Material receives every face of a batch in one call.
class ShellFemEnergy final : public energy::MeshEnergy {

  public:
    // Arrays and stencil rows must match the map's face and vertex order.
    // Required fixed Fp per face transforms natural forms; use identity matrices for no plasticity.
    // Original positions, area weights and thickness remain unchanged.
    ShellFemEnergy(std::shared_ptr<const ShellBatchMaterial> material, const dof::StencilDofMap& displacement_map,
                   std::vector<TriangleMidEdgeRestData> rest_data, std::vector<double> thickness,
                   std::vector<Eigen::VectorXd> theta, const std::vector<Eigen::Matrix3d>& Fp, int batch_size = 256)
        : energy::MeshEnergy(displacement_map, batch_size), material_(std::move(material)),
          rest_data_(std::move(rest_data)), thickness_(std::move(thickness)), theta_(std::move(theta)) {
        if (!material_ || material_->num_parameters() < 0)
            throw std::invalid_argument("ShellFemEnergy: invalid material");
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        if (displacement_map.dofs_per_vertex() != 3 || rest_data_.size() != count || thickness_.size() != count ||
            theta_.size() != count)
            throw std::invalid_argument("ShellFemEnergy: incompatible map or face counts");
        if (Fp.size() != count)
            throw std::invalid_argument("ShellFemEnergy: require one Fp per face");
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            const auto index = static_cast<std::size_t>(e);
            auto& rest = rest_data_[index];
            if (displacement_map.vertex_indices(e).size() != static_cast<std::size_t>(rest.X_local.rows()) ||
                !std::isfinite(rest.sqrt_det_a_bar) || rest.sqrt_det_a_bar <= 0)
                throw std::invalid_argument("ShellFemEnergy: incompatible reference stencil or area");
            const auto natural = detail::shell_plastic_geometry(rest.X_local, rest.opposite, Fp[index]);
            // Preserve original positions and area: only the natural metric and curvature change.
            rest.a_bar = natural.a;
            rest.b_bar = natural.b;
        }

        std::vector<int> ids(count);
        for (std::size_t e = 0; e < count; ++e)
            ids[e] = static_cast<int>(e);
        validate(ids, energy::BatchMatrix::Zero(static_cast<Eigen::Index>(count), 0));
    }

  private:
    Vec eval_batch(std::span<const int> ids, const energy::BatchMatrix& inputs, energy::BatchMatrix* gradients,
                   energy::BatchMatrix* hessians) const override {
        const Eigen::Index count = inputs.rows(), n = inputs.cols();
        Vec values(count);
        if (gradients)
            gradients->resize(count, n);
        if (hessians)
            hessians->resize(count * n, n);

        const auto states = this->states(ids, inputs, gradients != nullptr, hessians != nullptr);
        ShellMaterialBatch batch = detail::make_shell_batch(count, material_->num_parameters());
        for (Eigen::Index b = 0; b < count; ++b)
            detail::set_shell_point(batch, b, states[b].inputs);
        Vec W;
        Eigen::Matrix<double, 6, Eigen::Dynamic> dW;
        Eigen::Matrix<double, 36, Eigen::Dynamic> d2W;
        ShellMaterialBatchOutputs out{&W, (gradients || hessians) ? &dW : nullptr, hessians ? &d2W : nullptr};
        material_->eval(batch, out);
        detail::check_shell_batch_outputs(out, count, material_->num_parameters());
        auto integrate_row = [&](Eigen::Index b) {
            auto g = detail::batch_vector_view(gradients, b);
            auto H = detail::batch_matrix_view(hessians, b, n);
            values[b] = detail::finish_shell_element(
                states[b], W[b], Eigen::Map<const Vec>(out.dW_dab ? dW.col(b).data() : nullptr, out.dW_dab ? 6 : 0),
                Eigen::Map<const Eigen::MatrixXd>(hessians ? d2W.col(b).data() : nullptr, hessians ? 6 : 0,
                                                  hessians ? 6 : 0),
                g ? &*g : nullptr, H ? &*H : nullptr);
        };
        if (count == 1)
            integrate_row(0);
        else
            tbb::parallel_for(Eigen::Index(0), count, integrate_row);

        return values;
    }
    std::vector<detail::ShellElementState> states(std::span<const int> ids, const energy::BatchMatrix& inputs,
                                                  bool need_gradient, bool need_hessian) const {
        std::vector<detail::ShellElementState> result(ids.size());
        const Eigen::Index n = inputs.cols();
        auto gather_row = [&](std::size_t b) {
            const auto index = static_cast<std::size_t>(ids[b]);
            const auto& rest = rest_data_[index];
            const Eigen::MatrixXd u = n == 0 ? Eigen::MatrixXd::Zero(rest.X_local.rows(), 3)
                                             : Eigen::MatrixXd(inputs.row(b).reshaped<Eigen::RowMajor>(n / 3, 3));
            result[b] = detail::shell_element_state(rest, u, thickness_[index], theta_[index],
                                                    material_->num_parameters(), need_gradient, need_hessian);
        };
        if (ids.size() == 1) {
            gather_row(0);
        } else {
            // Outer cancellation must not leave partially initialized states.
            tbb::task_group_context context(tbb::task_group_context::isolated);
            tbb::parallel_for(std::size_t(0), ids.size(), gather_row, context);
        }
        return result;
    }
    // Throws when the material rejects these displacements (zero columns mean the rest pose).
    void validate(std::span<const int> ids, const energy::BatchMatrix& inputs) const {
        const auto states = this->states(ids, inputs, false, false);
        ShellMaterialBatch batch =
            detail::make_shell_batch(static_cast<Eigen::Index>(ids.size()), material_->num_parameters());
        for (std::size_t b = 0; b < ids.size(); ++b)
            detail::set_shell_point(batch, static_cast<Eigen::Index>(b), states[b].inputs);
        Vec W;
        ShellMaterialBatchOutputs out{&W};
        material_->eval(batch, out);
        detail::check_shell_batch_outputs(out, batch.size(), material_->num_parameters());
    }
    Vec eval_feasible_step_batch(std::span<const int> ids, const energy::BatchMatrix& inputs,
                                 const energy::BatchMatrix& deltas) const override {
        Vec steps(ids.size());
        const Eigen::Index n = inputs.cols();
        validate(ids, inputs);
        for (Eigen::Index b = 0; b < inputs.rows(); ++b) {
            const int e = ids[b];
            const Eigen::MatrixXd u = inputs.row(b).reshaped<Eigen::RowMajor>(n / 3, 3);
            const Eigen::MatrixXd du = deltas.row(b).reshaped<Eigen::RowMajor>(n / 3, 3);

            const auto index = static_cast<std::size_t>(e);
            const auto& rest = rest_data_[index];

            steps[b] = eval_triangle_mid_edge_feasible_step(rest.X_local + u, du, rest.opposite);
        }
        return steps;
    }
    std::shared_ptr<const ShellBatchMaterial> material_;
    std::vector<TriangleMidEdgeRestData> rest_data_;
    std::vector<double> thickness_;
    std::vector<Eigen::VectorXd> theta_;
};

} // namespace litepgo::fem
