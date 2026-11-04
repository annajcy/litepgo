// Assemble subclass residuals and fit two shared parameters.
#include "litepgo/dof/stencil_dof_map.hpp"
#include "litepgo/energy/response/mesh_response.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include <iostream>
#include <stdexcept>

using namespace litepgo;

class LinearResidual final : public energy::MeshResponse {
  public:
    LinearResidual(const dof::StencilDofMap& inputs, const dof::StencilDofMap& outputs, Eigen::Matrix2d matrix,
                   energy::BatchMatrix observations)
        : MeshResponse(inputs, outputs, 2), A(std::move(matrix)), observed(std::move(observations)) {}

  private:
    energy::BatchMatrix eval_batch(std::span<const int> ids, const energy::BatchMatrix& x,
                                   energy::BatchMatrix* J) const override {
        energy::BatchMatrix values = x * A.transpose();
        if (J)
            J->resize(2 * x.rows(), 2);
        for (Eigen::Index b = 0; b < x.rows(); ++b) {
            values.row(b) -= observed.row(ids[static_cast<std::size_t>(b)]);
            if (J)
                J->middleRows(2 * b, 2) = A;
        }
        return values;
    }
    energy::BatchMatrix eval_jvp_batch(std::span<const int>, const energy::BatchMatrix&,
                                       const energy::BatchMatrix& dx) const override {
        return dx * A.transpose();
    }
    energy::BatchMatrix eval_vjp_batch(std::span<const int>, const energy::BatchMatrix&,
                                       const energy::BatchMatrix& gy) const override {
        return gy * A;
    }
    Eigen::Matrix2d A;
    energy::BatchMatrix observed;
};

int main() {
    const dof::StencilDofMap inputs(2, {{0, 1}, {0, 1}, {0, 1}}, 1);
    const dof::StencilDofMap outputs(3, {{0, 1}, {1, 2}, {0, 2}}, 1);
    Eigen::Matrix2d A;
    A << 1, 2, 3, -1;
    const Vec truth = Eigen::Vector2d(2, 3);
    const energy::BatchMatrix observed = (A * truth).transpose().replicate(3, 1);
    LinearResidual response(inputs, outputs, A, observed);
    const Vec x = Eigen::Vector2d(.7, 1.2);
    // Independently sum the local rows assigned to each global output.
    Eigen::Matrix<double, 3, 2> J;
    J << 2, 4, 4, 1, 6, -2;
    const Vec v = Eigen::Vector2d(.3, -.8), w = Eigen::Vector3d(.2, -.5, .7);
    const Vec jv = response.eval_jvp(x, v), jtw = response.eval_vjp(x, w);
    const double h = 1e-5;
    const Vec numerical = (response.eval_value(x + h * v) - response.eval_value(x - h * v)) / (2 * h);
    if (!response.eval_value(x).isApprox(J * (x - truth)) || !Eigen::MatrixXd(response.eval_jacobian(x)).isApprox(J) ||
        !jv.isApprox(J * v) || !jtw.isApprox(J.transpose() * w) || std::abs(w.dot(jv) - v.dot(jtw)) > 1e-12 ||
        (jv - numerical).norm() > 1e-9)
        throw std::runtime_error("Assembly or product check failed");
    solver::GaussNewtonOptions options;
    options.gradient_tolerance = 1e-10;
    const auto result = solver::gauss_newton_solve(response, x, options);
    if (result.status != solver::GaussNewtonStatus::Converged || (result.x - truth).norm() > 1e-8)
        throw std::runtime_error("Parameter fit failed");
    std::cout << "mesh response subclass: shared assembly, JVP/VJP and parameter fit verified\n";
}
