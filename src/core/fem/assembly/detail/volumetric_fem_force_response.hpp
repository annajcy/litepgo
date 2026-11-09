#pragma once
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/fem/assembly/basis_dof_transform.hpp"
#include "litepgo/fem/assembly/detail/batch_views.hpp"
#include "litepgo/fem/integration/volumetric_element.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_batch.hpp"
#include "litepgo/fem/response/deformation_gradient_sampling_response.hpp"
#include <array>
#include <concepts>
#include <memory>
#include <utility>
#include <vector>

namespace litepgo::fem {
namespace detail {
enum class ForceInput { Material, Plastic };
// Fixed displacement and geometry; the other material input is stored per element.
template <Basis3D UBasis, Quadrature3D Q, ForceInput Input>
class VolumetricFemForceResponse final : public energy::MeshResponse {
    static constexpr int output_size = 3 * UBasis::num_basis_functions;

  public:
    template <class ParameterMap>
        requires(std::same_as<ParameterMap, dof::ConstantDofMap> || std::same_as<ParameterMap, dof::ElementDofMap>)
    VolumetricFemForceResponse(std::shared_ptr<const VolumetricBatchMaterial> material,
                               const ParameterMap& parameter_map, const dof::VertexDofMap& displacement_map,
                               std::vector<std::shared_ptr<energy::Response>> rest_position_fields, Vec displacement,
                               std::vector<std::shared_ptr<energy::Response>> fixed_fields, int batch_size = 256)
        : energy::MeshResponse(parameter_map, displacement_map, batch_size), material_(std::move(material)) {
        if (!material_ || material_->num_parameters() < 0)
            throw std::invalid_argument("VolumetricFemForceResponse: invalid material");
        const int parameters = material_->num_parameters();
        input_size_ = Input == ForceInput::Plastic ? 9 : parameters;
        const int fixed_size = Input == ForceInput::Plastic ? parameters : 9;
        if (parameter_map.num_elements() != displacement_map.num_elements() || batch_size <= 0)
            throw std::invalid_argument("VolumetricFemForceResponse: incompatible element count or batch size");
        const int width = [&] {
            if constexpr (std::same_as<ParameterMap, dof::ConstantDofMap>)
                return parameter_map.num_dofs();
            else
                return parameter_map.dofs_per_element();
        }();
        if (width != input_size_)
            throw std::invalid_argument("VolumetricFemForceResponse: invalid parameter count");
        const auto count = static_cast<std::size_t>(displacement_map.num_elements());
        if (displacement_map.dofs_per_vertex() != BasisDofTransform<UBasis>::dofs_per_vertex ||
            displacement.size() != displacement_map.num_dofs() || !displacement.allFinite() ||
            rest_position_fields.size() != count || fixed_fields.size() != count)
            throw std::invalid_argument("VolumetricFemForceResponse: incompatible map, displacement or responses");
        fixed_.resize(count);
        deformation_.resize(count);
        deformation_jacobians_.reserve(count);
        integration_.reserve(count);
        transforms_.reserve(count);
        for (int e = 0; e < displacement_map.num_elements(); ++e) {
            if (!rest_position_fields[e] || !fixed_fields[e] || displacement_map.dof_indices(e).size() != output_size ||
                fixed_fields[e]->num_inputs() != 3 || fixed_fields[e]->num_outputs() != fixed_size)
                throw std::invalid_argument("VolumetricFemForceResponse: incompatible spatial response or basis");
            transforms_.emplace_back(*rest_position_fields[e]);
            deformation_jacobians_.emplace_back();
            auto sampling = deformation_gradient_sampling_response<UBasis>(*rest_position_fields[e], Q::points());
            const Eigen::MatrixXd A = sampling.evaluate(Vec::Zero(output_size))->jacobian();
            for (int q = 0; q < Q::num_points; ++q)
                deformation_jacobians_.back()[q] = A.middleRows(9 * q, 9);
            integration_.emplace_back(*rest_position_fields[e]);
            const Eigen::MatrixXd coefficients =
                transforms_.back().to_coefficients(displacement_map.gather(e, displacement));
            const Vec flattened = coefficients.reshaped<Eigen::RowMajor>();
            const Vec values = sampling.evaluate(flattened)->value();
            for (int q = 0; q < Q::num_points; ++q) {
                deformation_[e][q] = values.segment<9>(9 * q).reshaped(3, 3);
                fixed_[e][q] = fixed_fields[e]->evaluate(Q::points().row(q).transpose())->value();
            }
        }
    }

  private:
    energy::BatchMatrix eval_jvp_batch(std::span<const int> ids, const energy::BatchMatrix& x,
                                       const energy::BatchMatrix& d) const override {
        check_batch_product(ids, x, d, false);
        return evaluate_batch(ids, x, nullptr, &d, false);
    }
    energy::BatchMatrix eval_vjp_batch(std::span<const int> ids, const energy::BatchMatrix& x,
                                       const energy::BatchMatrix& d) const override {
        check_batch_product(ids, x, d, true);
        return evaluate_batch(ids, x, nullptr, &d, true);
    }
    energy::BatchMatrix eval_batch(std::span<const int> ids, const energy::BatchMatrix& coefficients,
                                   energy::BatchMatrix* jacobian) const override {
        return evaluate_batch(ids, coefficients, jacobian);
    }
    energy::BatchMatrix evaluate_batch(std::span<const int> ids, const energy::BatchMatrix& coefficients,
                                       energy::BatchMatrix* jacobian, const energy::BatchMatrix* direction = nullptr,
                                       bool transpose = false) const {
        energy::BatchMatrix values(ids.size(), output_size);
        energy::BatchMatrix product =
            energy::BatchMatrix::Zero(direction ? ids.size() : 0, transpose ? input_size_ : output_size);
        if (jacobian)
            jacobian->resize(output_size * ids.size(), input_size_);

        const auto points = static_cast<Eigen::Index>(ids.size()) * Q::num_points;
        VolumetricMaterialBatch batch;
        batch.F.resize(9, points);
        batch.Fp.resize(9, points);
        batch.theta.resize(material_->num_parameters(), points);
        for (std::size_t b = 0; b < ids.size(); ++b) {
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Index point = static_cast<Eigen::Index>(b) * Q::num_points + q;
                batch.F.col(point) = deformation_[ids[b]][q].reshaped();
                if constexpr (Input == ForceInput::Plastic) {
                    batch.Fp.col(point) = coefficients.row(b).transpose();
                    batch.theta.col(point) = fixed_[ids[b]][q];
                } else {
                    batch.Fp.col(point) = fixed_[ids[b]][q];
                    batch.theta.col(point) = coefficients.row(b).transpose();
                }
            }
        }
        Eigen::Matrix<double, 9, Eigen::Dynamic> P;
        Eigen::MatrixXd dP_dtheta;
        Eigen::Matrix<double, 81, Eigen::Dynamic> dP_dFp;
        VolumetricMaterialBatchOutputs out{nullptr, &P};
        if (jacobian || direction) {
            if constexpr (Input == ForceInput::Plastic)
                out.d2W_dF_dFp = &dP_dFp;
            else
                out.d2W_dF_dtheta = &dP_dtheta;
        }
        material_->eval(batch, out);
        check_volumetric_batch_outputs(out, points, material_->num_parameters());
        for (std::size_t b = 0; b < ids.size(); ++b)
            eval_row(values, jacobian, b, ids[b], [&](int q) {
                const Eigen::Index point = static_cast<Eigen::Index>(b) * Q::num_points + q;
                const double* mixed = !(jacobian || direction)       ? nullptr
                                      : Input == ForceInput::Plastic ? dP_dFp.col(point).data()
                                                                     : dP_dtheta.col(point).data();
                return std::pair{Eigen::Map<const Eigen::Matrix3d>(P.col(point).data()),
                                 Eigen::Map<const Eigen::MatrixXd>(mixed, 9, mixed ? input_size_ : 0)};
            });

        if (direction) {
            if (!values.allFinite())
                throw std::domain_error("VolumetricFemForceResponse: non-finite force");
            for (std::size_t b = 0; b < ids.size(); ++b) {
                const int e = ids[b];
                Vec d = direction->row(b).transpose();
                // All supported DOF transforms are diagonal, so T and T^T coincide.
                if (transpose)
                    transforms_[e].pullback_response_into(StridedVecRef(d), nullptr);
                Vec result = Vec::Zero(transpose ? input_size_ : output_size);
                for (int q = 0; q < Q::num_points; ++q) {
                    const auto point = b * Q::num_points + q;
                    const double* ptr =
                        Input == ForceInput::Plastic ? dP_dFp.col(point).data() : dP_dtheta.col(point).data();
                    const Eigen::Map<const Eigen::MatrixXd> M(ptr, 9, input_size_);
                    const auto& B = deformation_jacobians_[e][q];
                    const double w = integration_[e].point(q).w_dV;
                    if (transpose) {
                        const Vec a = B * d;
                        result.noalias() += w * M.transpose() * a;
                    } else {
                        const Vec a = M * d;
                        result.noalias() += w * B.transpose() * a;
                    }
                }
                if (!transpose)
                    transforms_[e].pullback_response_into(StridedVecRef(result), nullptr);
                product.row(b) = result.transpose();
            }
        }
        return direction ? product : values;
    }

    // point_material(q) -> (dW/dF, d vec(dW/dF)/d active); the second is read only for Jacobians.
    template <class PointMaterial>
    void eval_row(energy::BatchMatrix& values, energy::BatchMatrix* jacobian, std::size_t b, int e,
                  PointMaterial&& point_material) const {
        StridedVecRef value(values.row(b).transpose());
        auto J = batch_matrix_view(jacobian, b, output_size);
        value.setZero();
        if (J)
            J->setZero();
        for (int q = 0; q < Q::num_points; ++q) {
            const auto [stress, mixed] = point_material(q);
            const auto& B = deformation_jacobians_[e][q];
            const double weight = integration_[e].point(q).w_dV;
            if (stress.rows() != 3 || stress.cols() != 3)
                throw std::logic_error("VolumetricFemForceResponse: invalid material stress dimensions");
            if (!stress.allFinite())
                throw std::domain_error("VolumetricFemForceResponse: non-finite material stress");
            value.noalias() += weight * B.transpose() * stress.reshaped();
            if (J) {
                if (mixed.rows() != 9 || mixed.cols() != input_size_)
                    throw std::logic_error("VolumetricFemForceResponse: invalid material mixed derivative");
                J->noalias() += weight * B.transpose() * mixed;
            }
        }
        transforms_[e].pullback_response_into(value, J ? &*J : nullptr);
    }

    std::shared_ptr<const VolumetricBatchMaterial> material_;
    int input_size_ = 0;
    std::vector<std::array<Vec, Q::num_points>> fixed_;
    std::vector<std::array<Eigen::Matrix3d, Q::num_points>> deformation_;
    std::vector<std::array<Eigen::MatrixXd, Q::num_points>> deformation_jacobians_;
    std::vector<VolumetricElement3D<Q>> integration_;
    std::vector<BasisDofTransform<UBasis>> transforms_;
};
} // namespace detail
} // namespace litepgo::fem
