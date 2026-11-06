#include "../../testing/element_formula_energy.hpp"
#include "../../testing/finite_difference.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/quadrature/2d/gauss_legendre_2d.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/fem/response/sampling_response.hpp"
#include <memory>

#include <Eigen/Geometry>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace litepgo::fem::test {

namespace {

const Eigen::Vector3d tangent1(2, 0.4, 1), tangent2(0.2, 3, -0.6);

std::shared_ptr<InterpolatedResponse<QuadHermiteBasis>> coordinates(int element) {
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(16, 3);
    for (int v = 0; v < 4; ++v) {
        const auto xi = QuadHermiteBasis::coefficient_info(4 * v).point;
        coefficients.row(4 * v) = ((element + xi.x()) * tangent1 + xi.y() * tangent2).transpose();
        coefficients.row(4 * v + 1) = tangent1.transpose();
        coefficients.row(4 * v + 2) = tangent2.transpose();
    }
    return std::make_shared<InterpolatedResponse<QuadHermiteBasis>>(coefficients);
}

testing::ElementFormulaEnergy surface_energy() {
    const mesh::QuadTopology topology(6, {{0, 1, 2, 3}, {1, 4, 5, 2}});
    const dof::VertexDofMap map(topology, 12);
    const std::array<std::shared_ptr<energy::Response>, 2> X_fields{coordinates(0), coordinates(1)};
    return testing::ElementFormulaEnergy(
        map, [X_fields](int e, const Eigen::MatrixXd& x_local, Vec* g, Eigen::MatrixXd* H) {
            const Vec c = x_local.reshaped<Eigen::RowMajor>();
            if (g)
                g->setZero(c.size());
            if (H)
                H->setZero(c.size(), c.size());
            using Q = GaussLegendre2D<4>;
            double result = 0;
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Vector2d xi = Q::points().row(q).transpose();
                const Eigen::MatrixXd derivative = X_fields[static_cast<std::size_t>(e)]->eval_jacobian(xi);
                const Eigen::Vector3d a = derivative.col(0), b = derivative.col(1);
                const Eigen::MatrixXd points = xi.transpose();
                auto value = value_sampling_response<QuadHermiteBasis>(points, 3);
                auto gradient = gradient_sampling_response<QuadHermiteBasis>(points, 3);
                auto hessian = hessian_sampling_response<QuadHermiteBasis>(points, 3);
                const Eigen::MatrixXd A = value.eval_jacobian(c), B = gradient.eval_jacobian(c),
                                      C = hessian.eval_jacobian(c);
                const Eigen::MatrixXd tangent =
                    Q::weights()[q] * a.cross(b).norm() * (A.transpose() * A + B.transpose() * B + C.transpose() * C);
                result += 0.5 * c.dot(tangent * c);
                if (g)
                    *g += tangent * c;
                if (H)
                    *H += tangent;
            }
            return result;
        });
}

TEST(SurfaceIntegration, IntegratesReferenceAreaForThreeComponentFieldsOnTwoDimensionalElements) {
    auto energy = surface_energy();
    Vec x = Vec::Zero(energy.num_dofs());
    const Eigen::Vector3d translation(0.3, -0.4, 0.7);
    for (int v = 0; v < 6; ++v)
        x.segment<3>(12 * v) = translation;
    const double total_area = 2 * tangent1.cross(tangent2).norm();
    EXPECT_NEAR(energy.eval_value(x), 0.5 * translation.squaredNorm() * total_area, 1e-12);
    EXPECT_DOUBLE_EQ(energy.eval_value(Vec::Zero(x.size())), 0);
}

TEST(SurfaceIntegration, AssembledGradientAndHessianMatchFiniteDifferencesWithSharedDofs) {
    auto energy = surface_energy();
    Vec x(energy.num_dofs());
    for (int j = 0; j < x.size(); ++j)
        x[j] = 0.07 * std::sin(0.3 * (j + 1));
    energy::test::expect_derivatives_match(energy, x, 2e-7);
    const Eigen::MatrixXd H = energy.eval_hessian(x);
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-13));
    EXPECT_GT(H.block(12, 48, 12, 12).norm(), 0);
}

} // namespace
} // namespace litepgo::fem::test
