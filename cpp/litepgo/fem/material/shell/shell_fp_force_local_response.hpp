#pragma once
#include "litepgo/fem/detail/shell_force_data.hpp"
#include "litepgo/fem/detail/shell_plastic_geometry.hpp"
#include "litepgo/fem/field/field.hpp"
#include "litepgo/fem/local/local_response.hpp"
#include <utility>

namespace litepgo::fem {
namespace detail {
struct ShellFpForceData {
    ShellForceData current;
    Eigen::MatrixXd reference, dy_dFp;
    std::array<int, 3> opposite;
    ShellFpForceData(const TriangleMidEdgeRestData& rest, const Eigen::MatrixXd& u, double thickness,
                     Eigen::VectorXd theta)
        : current(rest, u, thickness), opposite(rest.opposite) {
        if (!theta.allFinite())
            throw std::invalid_argument("Shell Fp force: non-finite material parameters");
        current.inputs.theta = std::move(theta);
        reference = rest.X_local.rowwise() - rest.X_local.row(0);
        if (!reference.allFinite())
            throw std::domain_error("Shell Fp force: non-finite reference offsets");
        dy_dFp = Eigen::MatrixXd::Zero(3 * reference.rows(), 9);
        for (Eigen::Index i = 0; i < reference.rows(); ++i)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    dy_dFp(3 * i + k, k + 3 * l) = reference(i, l);
    }
};

// Null dFp means the nine column-major Fp entries themselves are active.
template <class Material>
void accumulate_shell_fp_force(const ShellFpForceData& data, const Eigen::VectorXd& packed, const Eigen::MatrixXd* dFp,
                               double weight, StridedVecRef value, StridedMatRef* J) {
    const int n = dFp ? static_cast<int>(dFp->cols()) : 9;
    if (packed.size() != 9 || !packed.allFinite() || !std::isfinite(weight) ||
        data.current.inputs.theta.size() != Material::num_parameters || value.size() != data.current.dab_du.cols() ||
        (J && (J->rows() != value.size() || J->cols() != n)) ||
        (dFp && (dFp->rows() != 9 || n <= 0 || !dFp->allFinite())))
        throw std::invalid_argument("Shell Fp force: invalid Fp, derivative, parameters, weight or output dimensions");
    const Eigen::Matrix3d Fp = packed.reshaped(3, 3);
    Eigen::MatrixXd natural_derivative;
    const auto natural = shell_plastic_geometry(data.reference, data.opposite, Fp, J ? &natural_derivative : nullptr);
    auto inputs = data.current.inputs;
    inputs.a_bar = natural.a;
    inputs.b_bar = natural.b;
    const Eigen::VectorXd stress = Material::eval_dW_dab(inputs);
    if (stress.size() != 6)
        throw std::logic_error("Shell Fp force: invalid material gradient dimensions");
    if (!stress.allFinite())
        throw std::domain_error("Shell Fp force: non-finite material gradient");
    value.noalias() += weight * (data.current.dab_du.transpose() * stress);
    if (J) {
        const Eigen::MatrixXd mixed = Material::eval_d2W_dab_dabarbbar(inputs);
        if (mixed.rows() != 6 || mixed.cols() != 6)
            throw std::logic_error("Shell Fp force: invalid natural-form mixed derivative dimensions");
        if (!mixed.allFinite())
            throw std::domain_error("Shell Fp force: non-finite natural-form mixed derivative");
        const Eigen::MatrixXd full = data.current.dab_du.transpose() * mixed * natural_derivative * data.dy_dFp;
        if (dFp)
            J->noalias() += weight * (full * *dFp);
        else
            J->noalias() += weight * full;
    }
    if (!value.allFinite() || (J && !J->allFinite()))
        throw std::domain_error("Shell Fp force: non-finite accumulation");
}
} // namespace detail

// Fixed original geometry, displacement, thickness and theta; borrows a nine-component Field2D for Fp.
// Density uses the original area measure. Fp transforms the entire reference stencil only.
template <class Material> class ShellFpForceLocalResponse2D final : public LocalResponse2D {
  public:
    ShellFpForceLocalResponse2D(const TriangleMidEdgeRestData& rest, const Eigen::MatrixXd& u_local, double thickness,
                                const Field2D& Fp, Eigen::VectorXd theta)
        : data_(rest, u_local, thickness, std::move(theta)), Fp_(Fp) {
        if (Fp_.num_components() != 9 || Fp_.num_dofs() <= 0 ||
            data_.current.inputs.theta.size() != Material::num_parameters)
            throw std::invalid_argument("ShellFpForceLocalResponse2D: incompatible Fp field or material parameters");
    }
    ShellFpForceLocalResponse2D(const TriangleMidEdgeRestData&, const Eigen::MatrixXd&, double, const Field2D&&,
                                Eigen::VectorXd) = delete;
    int num_inputs() const override {
        return Fp_.num_dofs();
    }
    int num_outputs() const override {
        return static_cast<int>(data_.current.dab_du.cols());
    }
    bool has_jacobian() const noexcept override {
        return true;
    }
    Eigen::VectorXd eval_value(const Eigen::Vector2d& xi) const override {
        Eigen::VectorXd value = Eigen::VectorXd::Zero(num_outputs());
        accumulate_into(xi, 1, StridedVecRef(value));
        return value;
    }
    Eigen::MatrixXd eval_jacobian(const Eigen::Vector2d& xi) const override {
        Eigen::VectorXd value = Eigen::VectorXd::Zero(num_outputs());
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(num_outputs(), num_inputs());
        StridedMatRef J(matrix);
        accumulate_into(xi, 1, StridedVecRef(value), &J);
        return matrix;
    }
    void accumulate_into(const Eigen::Vector2d& xi, double weight, StridedVecRef value,
                         StridedMatRef* J = nullptr) const override {
        if (!xi.allFinite() || !std::isfinite(weight) || value.size() != num_outputs() ||
            (J && (J->rows() != num_outputs() || J->cols() != num_inputs())))
            throw std::invalid_argument("ShellFpForceLocalResponse2D: invalid coordinates, weight or outputs");
        const Eigen::VectorXd packed = Fp_.eval_value(xi);
        Eigen::MatrixXd derivative;
        if (J) {
            derivative = Fp_.eval_dvalue_dcoeffs(xi);
            if (derivative.cols() != num_inputs())
                throw std::invalid_argument("ShellFpForceLocalResponse2D: invalid Fp coefficient derivative");
        }
        detail::accumulate_shell_fp_force<Material>(data_, packed, J ? &derivative : nullptr, weight, value, J);
    }

  private:
    detail::ShellFpForceData data_;
    const Field2D& Fp_;
};
} // namespace litepgo::fem
