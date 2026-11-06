#pragma once
#include "litepgo/fem/assembly/detail/batch_views.hpp"
#include <tbb/parallel_for.h>

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/mesh_energy.hpp"
#include "litepgo/fem/assembly/basis_dof_transform.hpp"
#include "litepgo/fem/assembly/dof_derivatives.hpp"
#include "litepgo/fem/integration/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_batch.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"
#include "litepgo/fem/response/sampling_response.hpp"
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace litepgo::fem {

// Geometry and fixed material responses are sampled at construction; rebuild after changing them.
// Material is either a static pointwise material (NeoHookean) or a VolumetricBatchMaterial instance,
// which receives every quadrature point of a MeshEnergy batch in one call. Both share integration.
template <class Material, Basis3D UBasis, Quadrature3D Q> class VolumetricFemEnergy final : public energy::MeshEnergy {
    static constexpr bool batched = VolumetricBatchMaterial<Material>;

  public:
    // One spatial response (input xi in R3) per element, in displacement-map order.
    VolumetricFemEnergy(const dof::VertexDofMap& displacement_map,
                        const std::vector<std::shared_ptr<energy::Response>>& rest_positions,
                        const std::vector<std::shared_ptr<energy::Response>>& Fp,
                        const std::vector<std::shared_ptr<energy::Response>>& theta, int batch_size = 256,
                        Material material = Material())
        : energy::MeshEnergy(displacement_map, batch_size), material_(std::move(material)) {
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        const int parameters = detail::volumetric_num_parameters(material_);
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
        if constexpr (!batched) {
            tbb::parallel_for(Eigen::Index(0), count, [&](Eigen::Index b) {
                const auto e = static_cast<std::size_t>(ids[b]);
                const Vec c = local_coefficients(e, inputs.row(b));
                integrate(b, [&](int q, bool need_gradient, bool need_hessian) {
                    const auto input = material_inputs(e, q, c);
                    return std::tuple{Material::eval_W(input),
                                      need_gradient ? Material::eval_dW_dF(input) : Eigen::Matrix3d(),
                                      need_hessian ? Material::eval_d2W_dF2(input) : Eigen::Matrix<double, 9, 9>()};
                });
            });
        } else {
            const VolumetricMaterialBatch batch = gather(ids, inputs, nullptr);
            Vec W;
            Eigen::Matrix<double, 9, Eigen::Dynamic> P;
            Eigen::Matrix<double, 81, Eigen::Dynamic> A;
            VolumetricMaterialBatchOutputs out{&W, gradients ? &P : nullptr, hessians ? &A : nullptr};
            material_.eval(batch, out);
            detail::check_volumetric_batch_outputs(out, batch.size(), detail::volumetric_num_parameters(material_));
            tbb::parallel_for(Eigen::Index(0), count, [&](Eigen::Index b) {
                integrate(b, [&](int q, bool need_gradient, bool need_hessian) {
                    const Eigen::Index point = b * Q::num_points + q;
                    return std::tuple{
                        W[point], Eigen::Map<const Eigen::Matrix3d>(need_gradient ? P.col(point).data() : nullptr),
                        Eigen::Map<const Eigen::Matrix<double, 9, 9>>(need_hessian ? A.col(point).data() : nullptr)};
                });
            });
        }
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
        batch.theta.resize(detail::volumetric_num_parameters(material_), count * Q::num_points);
        if (dF)
            dF->resize(9, count * Q::num_points);
        tbb::parallel_for(Eigen::Index(0), count, [&](Eigen::Index b) {
            const auto e = static_cast<std::size_t>(ids[b]);
            const Vec c = local_coefficients(e, inputs.row(b));
            const Vec dc = deltas ? local_coefficients(e, deltas->row(b)) : Vec();
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Index point = b * Q::num_points + q;
                const auto input = material_inputs(e, q, c);
                batch.F.col(point) = input.F.reshaped();
                batch.Fp.col(point) = input.Fp.reshaped();
                batch.theta.col(point) = input.theta;
                if (dF) {
                    dF->col(point) = deformation_jacobians_[e][q] * dc;
                    if (!dF->col(point).allFinite())
                        throw std::domain_error("VolumetricFemEnergy: non-finite input increment");
                }
            }
        });
        return batch;
    }

    Vec eval_feasible_step_batch(std::span<const int> ids, const energy::BatchMatrix& inputs,
                                 const energy::BatchMatrix& deltas) const override {
        Vec steps(ids.size());
        if constexpr (batched) {
            Eigen::Matrix<double, 9, Eigen::Dynamic> dF;
            const VolumetricMaterialBatch batch = gather(ids, inputs, &deltas, &dF);
            const Vec bounds = material_.eval_feasible_step(batch, dF);
            if (bounds.size() != batch.size())
                throw std::logic_error("VolumetricFemEnergy: invalid batch feasible-step dimensions");
            for (Eigen::Index b = 0; b < inputs.rows(); ++b)
                steps[b] = integration_[static_cast<std::size_t>(ids[b])].max_feasible_step(
                    [&](int q, const auto&) { return bounds[b * Q::num_points + q]; });
        } else {
            for (Eigen::Index b = 0; b < inputs.rows(); ++b) {
                const auto index = static_cast<std::size_t>(ids[b]);
                const Vec flattened = local_coefficients(index, inputs.row(b));
                const Vec delta = local_coefficients(index, deltas.row(b));
                steps[b] = integration_[index].max_feasible_step([&](int q, const auto&) {
                    const auto input = material_inputs(index, q, flattened);
                    const Vec delta_F = deformation_jacobians_[index][q] * delta;
                    if (!delta_F.allFinite())
                        throw std::domain_error("VolumetricFemEnergy: non-finite input increment");
                    const VolumetricMaterialInputs direction{delta_F.reshaped(3, 3), Eigen::Matrix3d::Zero(),
                                                             Vec::Zero(input.theta.size())};
                    return Material::eval_feasible_step(input, direction);
                });
            }
        }
        return steps;
    }
    VolumetricMaterialInputs material_inputs(std::size_t e, int q, const Vec& coefficients) const {
        const Vec increment = deformation_jacobians_[e][q] * coefficients;
        const Eigen::Matrix3d F = Eigen::Matrix3d::Identity() + increment.reshaped(3, 3);
        if (!F.allFinite())
            throw std::domain_error("VolumetricFemEnergy: non-finite deformation gradient");
        return {F, Fp_[e][q], theta_[e][q]};
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

    Material material_;
    std::vector<std::array<Eigen::MatrixXd, Q::num_points>> deformation_jacobians_;
    std::vector<std::array<Eigen::Matrix3d, Q::num_points>> Fp_;
    std::vector<std::array<Vec, Q::num_points>> theta_;
    std::vector<VolumetricElement3D<Q>> integration_;
    std::vector<BasisDofTransform<UBasis>> transforms_;
};

} // namespace litepgo::fem
