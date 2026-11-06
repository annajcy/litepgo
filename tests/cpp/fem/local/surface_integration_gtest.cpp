#include "../../testing/element_formula_energy.hpp"
#include "../../testing/finite_difference.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/basis/2d/quad_hermite_basis.hpp"
#include "litepgo/fem/local/local_energy.hpp"
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

static_assert(std::is_same_v<LocalEnergy3D, LocalEnergyT<3>>);
static_assert(std::is_same_v<LocalEnergy2D, LocalEnergyT<2>>);

// Sampled value, gradient and Hessian are affine functions of the active coefficients.
class QuadraticSamplingEnergy final : public LocalEnergy2D {
  public:
    explicit QuadraticSamplingEnergy(const Eigen::MatrixXd& C)
        : coefficients_(C.reshaped<Eigen::RowMajor>()), components_(static_cast<int>(C.cols())) {}
    int num_dofs() const override {
        return static_cast<int>(coefficients_.size());
    }
    bool has_gradient() const noexcept override {
        return true;
    }
    bool has_hessian() const noexcept override {
        return true;
    }
    double eval_value(const Eigen::Vector2d& xi) const override {
        return 0.5 * coefficients_.dot(eval_hessian(xi) * coefficients_);
    }
    Vec eval_gradient(const Eigen::Vector2d& xi) const override {
        return eval_hessian(xi) * coefficients_;
    }
    Eigen::MatrixXd eval_hessian(const Eigen::Vector2d& xi) const override {
        const Eigen::MatrixXd points = xi.transpose();
        auto value = value_sampling_response<QuadHermiteBasis>(points, components_);
        auto gradient = gradient_sampling_response<QuadHermiteBasis>(points, components_);
        auto hessian = hessian_sampling_response<QuadHermiteBasis>(points, components_);
        const Eigen::MatrixXd A = value.eval_jacobian(coefficients_);
        const Eigen::MatrixXd B = gradient.eval_jacobian(coefficients_);
        const Eigen::MatrixXd C = hessian.eval_jacobian(coefficients_);
        return A.transpose() * A + B.transpose() * B + C.transpose() * C;
    }

  private:
    Vec coefficients_;
    int components_;
};

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
            const QuadraticSamplingEnergy local(x_local.reshaped<Eigen::RowMajor>(16, 3));
            if (g)
                g->setZero(local.num_dofs());
            if (H)
                H->setZero(local.num_dofs(), local.num_dofs());
            using Q = GaussLegendre2D<4>;
            double result = 0;
            for (int q = 0; q < Q::num_points; ++q) {
                const Eigen::Vector2d xi = Q::points().row(q).transpose();
                const Eigen::MatrixXd derivative = X_fields[static_cast<std::size_t>(e)]->eval_jacobian(xi);
                const Eigen::Vector3d a = derivative.col(0), b = derivative.col(1);
                result += local.accumulate(xi, Q::weights()[q] * a.cross(b).norm(), g, H);
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

TEST(SurfaceIntegration, LocalEnergy2DAccumulatesAndValidatesThroughBaseReference) {
    auto field = std::make_shared<InterpolatedResponse<QuadHermiteBasis>>(Eigen::MatrixXd::Constant(16, 2, 0.3));
    const QuadraticSamplingEnergy concrete(Eigen::MatrixXd::Constant(16, 2, 0.3));
    const LocalEnergy2D& local = concrete;
    const Eigen::Vector2d xi(0.2, 0.3);
    const double weight = -0.7;
    Vec g = Vec::Constant(local.num_dofs(), 3);
    Eigen::MatrixXd H = Eigen::MatrixXd::Constant(local.num_dofs(), local.num_dofs(), 5);
    EXPECT_DOUBLE_EQ(local.accumulate(xi, weight, &g, &H), weight * local.eval_value(xi));
    EXPECT_TRUE(g.isApprox(Vec::Constant(local.num_dofs(), 3) + weight * local.eval_gradient(xi), 1e-13));
    EXPECT_TRUE(H.isApprox(
        Eigen::MatrixXd::Constant(local.num_dofs(), local.num_dofs(), 5) + weight * local.eval_hessian(xi), 1e-13));
    EXPECT_DOUBLE_EQ(local.accumulate(xi, weight, nullptr, nullptr), weight * local.eval_value(xi));
    const auto before_g = g;
    const auto before_H = H;
    EXPECT_THROW(local.accumulate(xi, std::numeric_limits<double>::infinity(), &g, &H), std::invalid_argument);
    EXPECT_THROW(local.accumulate(Eigen::Vector2d::Constant(std::numeric_limits<double>::quiet_NaN()), 1, &g, &H),
                 std::invalid_argument);
    EXPECT_EQ(g, before_g);
    EXPECT_EQ(H, before_H);
    Vec wrong_g = Vec::Zero(local.num_dofs() - 1);
    Eigen::MatrixXd wrong_H = Eigen::MatrixXd::Zero(local.num_dofs(), local.num_dofs() - 1);
    EXPECT_THROW(local.accumulate(xi, 1, &wrong_g, nullptr), std::invalid_argument);
    EXPECT_THROW(local.accumulate(xi, 1, nullptr, &wrong_H), std::invalid_argument);
}

struct ValueOnly2D final : LocalEnergy2D {
    int num_dofs() const override {
        return 1;
    }
    double eval_value(const Eigen::Vector2d& xi) const override {
        return xi.sum();
    }
};

TEST(SurfaceIntegration, LocalEnergy2DRetainsOptionalDerivativeCapabilities) {
    const ValueOnly2D local;
    const Eigen::Vector2d xi(0.2, 0.3);
    EXPECT_FALSE(local.has_gradient());
    EXPECT_FALSE(local.has_hessian());
    EXPECT_THROW(local.eval_gradient(xi), std::logic_error);
    EXPECT_THROW(local.eval_hessian(xi), std::logic_error);
    EXPECT_DOUBLE_EQ(local.accumulate(xi, 2, nullptr, nullptr), 1);
    Vec g = Vec::Zero(1);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 1);
    EXPECT_THROW(local.accumulate(xi, 1, &g, nullptr), std::logic_error);
    EXPECT_THROW(local.accumulate(xi, 1, nullptr, &H), std::logic_error);
}
} // namespace

} // namespace litepgo::fem::test
