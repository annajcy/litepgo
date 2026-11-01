#pragma once
#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/dof_derivatives.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"
#include <cmath>
#include <stdexcept>

namespace litepgo::fem {
namespace detail {
enum class ForceInput { Material, Plastic };

// Fixed F and dF/dcoeffs; differentiate the force density wrt one input field.
template <class Material, ForceInput Input> class VolumetricForceLocalResponse3D final : public LocalResponse3D {
  public:
    VolumetricForceLocalResponse3D(const Field3D& F, const Field3D& Fp, const Field3D& theta)
        : F_(F), Fp_(Fp), theta_(theta) {
        if (F.num_components() != 9 || Fp.num_components() != 9 || theta.num_components() != Material::num_parameters ||
            F.num_dofs() <= 0 || active_field().num_dofs() <= 0)
            throw std::invalid_argument("VolumetricForceLocalResponse3D: incompatible fields");
    }
    VolumetricForceLocalResponse3D(const Field3D&&, const Field3D&, const Field3D&) = delete;
    VolumetricForceLocalResponse3D(const Field3D&, const Field3D&&, const Field3D&) = delete;
    VolumetricForceLocalResponse3D(const Field3D&, const Field3D&, const Field3D&&) = delete;
    int num_inputs() const override {
        return active_field().num_dofs();
    }
    int num_outputs() const override {
        return F_.num_dofs();
    }
    bool has_jacobian() const noexcept override {
        return true;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector3d& xi) const override {
        Eigen::VectorXd value = Eigen::VectorXd::Zero(num_outputs());
        evaluate(xi, 1, &value, static_cast<Eigen::MatrixXd*>(nullptr), true);
        return value;
    }
    Eigen::MatrixXd eval_jacobian(const Eigen::Vector3d& xi) const override {
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(num_outputs(), num_inputs());
        evaluate(xi, 1, static_cast<Eigen::VectorXd*>(nullptr), &J, true);
        return J;
    }
    void accumulate(const Eigen::Vector3d& xi, double weight, Eigen::VectorXd& value,
                    Eigen::MatrixXd* jacobian = nullptr) const override {
        if (!xi.allFinite() || !std::isfinite(weight))
            throw std::invalid_argument("VolumetricForceLocalResponse3D: invalid coordinates or weight");
        evaluate(xi, weight, &value, jacobian, true);
    }
    void accumulate_for_integration(const Eigen::Vector3d& xi, double weight, Eigen::VectorXd& value,
                                    Eigen::MatrixXd* jacobian = nullptr) const override {
        evaluate(xi, weight, &value, jacobian, false);
    }

    void accumulate_into(const Eigen::Vector3d& xi, double weight, StridedVecRef value,
                         StridedMatRef* jacobian = nullptr) const override {
        evaluate(xi, weight, &value, jacobian, false);
    }

  private:
    const Field3D& active_field() const {
        if constexpr (Input == ForceInput::Plastic)
            return Fp_;
        else
            return theta_;
    }
    static constexpr int input_components = Input == ForceInput::Plastic ? 9 : Material::num_parameters;

    VolumetricMaterialInputs eval_inputs(const Eigen::Vector3d& xi) const {
        if (!xi.allFinite())
            throw std::invalid_argument("VolumetricForceLocalResponse3D: invalid coordinates");
        const auto F = F_.eval_value(xi), Fp = Fp_.eval_value(xi), theta = theta_.eval_value(xi);
        if (F.size() != 9 || Fp.size() != 9 || theta.size() != Material::num_parameters || !F.allFinite() ||
            !Fp.allFinite() || !theta.allFinite())
            throw std::invalid_argument("VolumetricForceLocalResponse3D: invalid material inputs");
        return {F.reshaped(3, 3), Fp.reshaped(3, 3), theta};
    }
    static Eigen::MatrixXd derivative(const Field3D& field, const Eigen::Vector3d& xi, int rows, bool check) {
        Eigen::MatrixXd D = field.eval_dvalue_dcoeffs(xi);
        if (D.rows() != rows || D.cols() != field.num_dofs() || (check && !D.allFinite()))
            throw std::invalid_argument("VolumetricForceLocalResponse3D: invalid field derivative");
        return D;
    }
    template <class V, class M>
    void evaluate(const Eigen::Vector3d& xi, double weight, V* value, M* J, bool check) const {
        if ((value && value->size() != num_outputs()) ||
            (J && (J->rows() != num_outputs() || J->cols() != num_inputs())))
            throw std::invalid_argument("VolumetricForceLocalResponse3D: invalid output dimensions");
        const auto inputs = eval_inputs(xi);
        const auto D_F = derivative(F_, xi, 9, check);
        if (value)
            add_gradient_wrt_dofs(D_F, Material::eval_dW_dF(inputs), weight, *value, false);
        if (J) {
            const auto D_input = derivative(active_field(), xi, input_components, check);
            const auto B = [&] {
                if constexpr (Input == ForceInput::Plastic)
                    return Material::eval_d2W_dF_dFp(inputs);
                else
                    return Material::eval_d2W_dF_dtheta(inputs);
            }();
            if (B.rows() != 9 || B.cols() != input_components)
                throw std::logic_error("VolumetricForceLocalResponse3D: invalid material mixed derivative");
            const Eigen::MatrixXd dP = B * D_input;
            J->noalias() += weight * (D_F.transpose() * dP);
        }
        if (check && ((value && !value->allFinite()) || (J && !J->allFinite())))
            throw std::domain_error("VolumetricForceLocalResponse3D: non-finite response");
    }
    const Field3D& F_;
    const Field3D& Fp_;
    const Field3D& theta_;
};
} // namespace detail
} // namespace litepgo::fem
