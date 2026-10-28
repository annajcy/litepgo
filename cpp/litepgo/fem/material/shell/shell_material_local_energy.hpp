#pragma once

#include "litepgo/fem/kinematics/triangle_mid_edge_kinematics.hpp"
#include "litepgo/fem/local/local_energy.hpp"
#include "litepgo/fem/material/shell/shell_material_inputs.hpp"

#include <Eigen/LU>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace litepgo {

// Build once per fixed reference face, using the same stencil and curvature rule as the current pose.
struct TriangleMidEdgeRestData {
    TriangleMidEdgeRestData(Eigen::MatrixXd X, std::array<int, 3> opposite_rows)
        : X_local(std::move(X)), opposite(opposite_rows) {
        const auto reference = eval_triangle_mid_edge_kinematics(X_local, opposite);
        a_bar = reference.a;
        b_bar = reference.b;
        sqrt_det_a_bar = std::sqrt(a_bar.determinant());
        if (!std::isfinite(sqrt_det_a_bar) || sqrt_det_a_bar <= 0)
            throw std::domain_error("TriangleMidEdgeRestData: unrepresentable reference area");
    }

    Eigen::MatrixXd X_local;
    std::array<int, 3> opposite;
    Eigen::Matrix2d a_bar;
    Eigen::Matrix2d b_bar;
    double sqrt_det_a_bar;
};

// Density per reference area; only row-packed stencil displacements are active.
// Owns snapshots of all inputs. Thickness and material parameters are constant on the face.
// Caller applies the complete quadrature/reference-area weight through accumulate.
template <class Material> class ShellMaterialLocalEnergy2D final : public LocalEnergy2D {
  public:
    ShellMaterialLocalEnergy2D(const TriangleMidEdgeRestData& rest, Eigen::MatrixXd u_local, double thickness,
                               Eigen::VectorXd theta)
        : x_local_(std::move(u_local)), opposite_(rest.opposite), a_bar_(rest.a_bar), b_bar_(rest.b_bar),
          thickness_(thickness), theta_(std::move(theta)) {
        if (rest.X_local.rows() < 3 || rest.X_local.rows() > 6 || rest.X_local.cols() != 3 ||
            x_local_.rows() != rest.X_local.rows() || x_local_.cols() != 3 || !x_local_.allFinite() ||
            !rest.X_local.allFinite() || !std::isfinite(thickness_) || thickness_ <= 0 ||
            theta_.size() != Material::num_parameters || !theta_.allFinite())
            throw std::invalid_argument("ShellMaterialLocalEnergy2D: invalid stencil, thickness, or parameters");
        x_local_ += rest.X_local;
        if (!x_local_.allFinite())
            throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite current positions");
    }

    int num_dofs() const override {
        return static_cast<int>(x_local_.size());
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }
    double eval_value(const Eigen::Vector2d& xi) const override {
        return eval_density(xi, nullptr, nullptr);
    }
    Eigen::VectorXd eval_gradient(const Eigen::Vector2d& xi) const override {
        Eigen::VectorXd gradient;
        eval_density(xi, &gradient, nullptr);
        return gradient;
    }
    Eigen::MatrixXd eval_hessian(const Eigen::Vector2d& xi) const override {
        Eigen::MatrixXd hessian;
        eval_density(xi, nullptr, &hessian);
        return hessian;
    }
    double accumulate(const Eigen::Vector2d& xi, double weight, Eigen::VectorXd* gradient,
                      Eigen::MatrixXd* hessian) const override {
        if (!std::isfinite(weight) || (gradient && gradient->size() != num_dofs()) ||
            (hessian && (hessian->rows() != num_dofs() || hessian->cols() != num_dofs())))
            throw std::invalid_argument("ShellMaterialLocalEnergy2D::accumulate: invalid weight or output dimensions");
        Eigen::VectorXd g;
        Eigen::MatrixXd H;
        const double W = eval_density(xi, gradient ? &g : nullptr, hessian ? &H : nullptr);
        const double result = weight * W;
        if (!std::isfinite(result))
            throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite weighted density");
        if (gradient) {
            *gradient += weight * g;
            if (!gradient->allFinite())
                throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite accumulated gradient");
        }
        if (hessian) {
            *hessian += weight * H;
            if (!hessian->allFinite())
                throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite accumulated Hessian");
        }
        return result;
    }

  private:
    double eval_density(const Eigen::Vector2d& xi, Eigen::VectorXd* gradient, Eigen::MatrixXd* hessian) const {
        if (!xi.allFinite())
            throw std::invalid_argument("ShellMaterialLocalEnergy2D requires finite parameter coordinates");
        Eigen::MatrixXd dab_dx;
        std::array<Eigen::MatrixXd, 6> d2ab_dx2;
        const auto current = eval_triangle_mid_edge_kinematics(
            x_local_, opposite_, (gradient || hessian) ? &dab_dx : nullptr, hessian ? &d2ab_dx2 : nullptr);
        const ShellMaterialInputs inputs{current.a, current.b, a_bar_, b_bar_, thickness_, theta_};
        const double W = Material::eval_W(inputs);
        if (!std::isfinite(W))
            throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite density");
        if (gradient || hessian) {
            const Eigen::VectorXd dW_dab = Material::eval_dW_dab(inputs);
            if (dW_dab.size() != 6)
                throw std::logic_error("ShellMaterialLocalEnergy2D: invalid material gradient");
            if (!dW_dab.allFinite())
                throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite material gradient");
            if (gradient) {
                *gradient = dab_dx.transpose() * dW_dab;
                if (!gradient->allFinite())
                    throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite gradient");
            }
            if (hessian) {
                const Eigen::MatrixXd d2W_dab2 = Material::eval_d2W_dab2(inputs);
                if (d2W_dab2.rows() != 6 || d2W_dab2.cols() != 6)
                    throw std::logic_error("ShellMaterialLocalEnergy2D: invalid material Hessian");
                if (!d2W_dab2.allFinite())
                    throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite material Hessian");
                *hessian = dab_dx.transpose() * d2W_dab2 * dab_dx;
                for (std::size_t i = 0; i < d2ab_dx2.size(); ++i)
                    *hessian += dW_dab[static_cast<Eigen::Index>(i)] * d2ab_dx2[i];
                if (!hessian->allFinite())
                    throw std::domain_error("ShellMaterialLocalEnergy2D: non-finite Hessian");
            }
        }
        return W;
    }

    Eigen::MatrixXd x_local_;
    std::array<int, 3> opposite_;
    Eigen::Matrix2d a_bar_;
    Eigen::Matrix2d b_bar_;
    double thickness_;
    Eigen::VectorXd theta_;
};

} // namespace litepgo
