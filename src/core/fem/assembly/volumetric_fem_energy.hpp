#pragma once
#include "litepgo/fem/assembly/detail/batch_views.hpp"
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/fem/assembly/basis_dof_transform.hpp"
#include "litepgo/fem/assembly/dof_derivatives.hpp"
#include "litepgo/fem/integration/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_batch.hpp"
#include "litepgo/fem/response/deformation_gradient_sampling_response.hpp"
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace litepgo::fem {

// Geometry and fixed material responses are sampled at construction; rebuild after changing them.
// Material receives every quadrature point of a MeshEnergy batch in one call.
template <Basis3D UBasis, Quadrature3D Q> class VolumetricFemEnergy final : public energy::MeshEnergy {

  public:
    // One spatial response (input xi in R3) per element, in displacement-map order.
    VolumetricFemEnergy(std::shared_ptr<const VolumetricBatchMaterial> material,
                        const dof::VertexDofMap& displacement_map,
                        const std::vector<std::shared_ptr<energy::Response>>& rest_positions,
                        const std::vector<std::shared_ptr<energy::Response>>& Fp,
                        const std::vector<std::shared_ptr<energy::Response>>& theta, int batch_size = 256)
        : energy::MeshEnergy(displacement_map, batch_size), material_(std::move(material)) {
        if (!material_ || material_->num_parameters() < 0)
            throw std::invalid_argument("VolumetricFemEnergy: invalid material");
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        const int parameters = material_->num_parameters();
        if (displacement_map.dofs_per_vertex() != BasisDofTransform<UBasis>::dofs_per_vertex ||
            rest_positions.size() != count || Fp.size() != count || theta.size() != count)
            throw std::invalid_argument("VolumetricFemEnergy: incompatible map or response counts");
        integration_.reserve(count);
        transforms_.reserve(count);
        deformation_jacobians_.reserve(count);
        Fp_.resize(count);
        theta_.resize(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            if (!rest_positions[e] || !Fp[e] || !theta[e] || Fp[e]->num_inputs() != 3 || theta[e]->num_inputs() != 3 ||
                Fp[e]->num_outputs() != 9 || theta[e]->num_outputs() != parameters)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible spatial responses");
            if (displacement_map.dof_indices(e).size() != 3 * UBasis::num_basis_functions)
                throw std::invalid_argument("VolumetricFemEnergy: incompatible displacement basis");
            integration_.emplace_back(*rest_positions[e]);
            transforms_.emplace_back(*rest_positions[e]);
            deformation_jacobians_.emplace_back();
            auto sampling = deformation_gradient_sampling_response<UBasis>(*rest_positions[e], Q::points());
            const Eigen::MatrixXd A = sampling.eval_jacobian(Vec::Zero(3 * UBasis::num_basis_functions));
            for (int q = 0; q < Q::num_points; ++q)
                deformation_jacobians_.back()[q] = A.middleRows(9 * q, 9);
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Vector3d xi = Q::points().row(q).transpose();
                const Vec plastic = Fp[e]->eval_value(xi);
                Fp_[e][q] = plastic.reshaped(3, 3);
                theta_[e][q] = theta[e]->eval_value(xi);
            }
        }
    }

  private:
    static constexpr int dofs_per_vertex = BasisDofTransform<UBasis>::dofs_per_vertex;

    // Physical DOFs of one batch row to flattened reference coefficients.
    Vec local_coefficients(std::size_t e, const Eigen::Ref<const Eigen::RowVectorXd>& row) const {
        const Eigen::Index n = row.size();
        Eigen::MatrixXd u = row.reshaped<Eigen::RowMajor>(n / dofs_per_vertex, dofs_per_vertex);
        const Eigen::MatrixXd c = transforms_[e].to_coefficients(std::move(u));
        return c.reshaped<Eigen::RowMajor>();
    }

    Vec eval_batch(std::span<const int> ids, const energy::BatchMatrix& inputs, energy::BatchMatrix* gradients,
                   energy::BatchMatrix* hessians) const override {
        const Eigen::Index count = inputs.rows(), n = inputs.cols();
        Vec values(count);
        if (gradients)
            gradients->resize(count, n);
        if (hessians)
            hessians->resize(count * n, n);
        // Integrates one element from point values; native and batch materials share it.
        auto integrate = [&](Eigen::Index b, auto&& point_material) {
            const auto e = static_cast<std::size_t>(ids[b]);
            auto g = detail::batch_vector_view(gradients, b);
            auto H = detail::batch_matrix_view(hessians, b, n);
            values[b] = eval_element(e, static_cast<int>(n), g ? &*g : nullptr, H ? &*H : nullptr, point_material);
            transforms_[e].pullback_into(g ? &*g : nullptr, H ? &*H : nullptr);
        };

        const VolumetricMaterialBatch batch = gather(ids, inputs, nullptr);
        Vec W;
        Eigen::Matrix<double, 9, Eigen::Dynamic> P;
        Eigen::Matrix<double, 81, Eigen::Dynamic> A;
        VolumetricMaterialBatchOutputs out{&W, gradients ? &P : nullptr, hessians ? &A : nullptr};
        material_->eval(batch, out);
        detail::check_volumetric_batch_outputs(out, batch.size(), material_->num_parameters());
        auto integrate_row = [&](Eigen::Index b) {
            integrate(b, [&](int q, bool need_gradient, bool need_hessian) {
                const Eigen::Index point = b * Q::num_points + q;
                return std::tuple{
                    W[point], Eigen::Map<const Eigen::Matrix3d>(need_gradient ? P.col(point).data() : nullptr),
                    Eigen::Map<const Eigen::Matrix<double, 9, 9>>(need_hessian ? A.col(point).data() : nullptr)};
            });
        };
        if (count == 1)
            integrate_row(0);
        else
            tbb::parallel_for(Eigen::Index(0), count, integrate_row);

        return values;
    }

    // Material inputs at every (row, point), point-major within rows; optionally the dF of deltas.
    VolumetricMaterialBatch gather(std::span<const int> ids, const energy::BatchMatrix& inputs,
                                   const energy::BatchMatrix* deltas,
                                   Eigen::Matrix<double, 9, Eigen::Dynamic>* dF = nullptr) const {
        const Eigen::Index count = inputs.rows();
        VolumetricMaterialBatch batch;
        batch.F.resize(9, count * Q::num_points);
        batch.Fp.resize(9, count * Q::num_points);
        batch.theta.resize(material_->num_parameters(), count * Q::num_points);
        if (dF)
            dF->resize(9, count * Q::num_points);
        auto gather_row = [&](Eigen::Index b) {
            const auto e = static_cast<std::size_t>(ids[b]);
            const Vec c = local_coefficients(e, inputs.row(b));
            const Vec dc = deltas ? local_coefficients(e, deltas->row(b)) : Vec();
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Index point = b * Q::num_points + q;
                batch.F.col(point).noalias() = deformation_jacobians_[e][q] * c;
                batch.F.col(point).reshaped(3, 3).diagonal().array() += 1;
                if (!batch.F.col(point).allFinite())
                    throw std::domain_error("VolumetricFemEnergy: non-finite deformation gradient");
                batch.Fp.col(point) = Fp_[e][q].reshaped();
                batch.theta.col(point) = theta_[e][q];
                if (dF) {
                    dF->col(point).noalias() = deformation_jacobians_[e][q] * dc;
                    if (!dF->col(point).allFinite())
                        throw std::domain_error("VolumetricFemEnergy: non-finite input increment");
                }
            }
        };
        if (count == 1) {
            gather_row(0);
        } else {
            // Outer batch failure must not cancel initialization of inputs consumed below.
            tbb::task_group_context context(tbb::task_group_context::isolated);
            tbb::parallel_for(Eigen::Index(0), count, gather_row, context);
        }
        return batch;
    }

    Vec eval_feasible_step_batch(std::span<const int> ids, const energy::BatchMatrix& inputs,
                                 const energy::BatchMatrix& deltas) const override {
        Vec steps(ids.size());

        Eigen::Matrix<double, 9, Eigen::Dynamic> dF;
        const VolumetricMaterialBatch batch = gather(ids, inputs, &deltas, &dF);
        const Vec bounds = material_->eval_feasible_step(batch, dF);
        if (bounds.size() != batch.size())
            throw std::logic_error("VolumetricFemEnergy: invalid batch feasible-step dimensions");
        for (Eigen::Index b = 0; b < inputs.rows(); ++b)
            steps[b] = integration_[static_cast<std::size_t>(ids[b])].max_feasible_step(
                [&](int q, const auto&) { return bounds[b * Q::num_points + q]; });

        return steps;
    }
    // point_material(q, need_gradient, need_hessian) -> (W, dW/dF, d2W/dF2); unrequested entries are unused.
    template <class PointMaterial>
    double eval_element(std::size_t e, int num_dofs, StridedVecRef* gradient, StridedMatRef* hessian,
                        PointMaterial&& point_material) const {
        return integration_[e].eval_into(
            num_dofs,
            [&](int q, const auto& point, StridedVecRef* g, StridedMatRef* H) {
                const auto [W, P, A] = point_material(q, g != nullptr, H != nullptr);
                const auto& derivative = deformation_jacobians_[e][q];
                if (g)
                    add_gradient_wrt_dofs(derivative, P, point.w_dV, *g, false);
                if (H)
                    add_hessian_wrt_dofs(derivative, A, point.w_dV, *H, false);
                return point.w_dV * W;
            },
            gradient, hessian);
    }

    std::shared_ptr<const VolumetricBatchMaterial> material_;
    std::vector<std::array<Eigen::MatrixXd, Q::num_points>> deformation_jacobians_;
    std::vector<std::array<Eigen::Matrix3d, Q::num_points>> Fp_;
    std::vector<std::array<Vec, Q::num_points>> theta_;
    std::vector<VolumetricElement3D<Q>> integration_;
    std::vector<BasisDofTransform<UBasis>> transforms_;
};

} // namespace litepgo::fem
