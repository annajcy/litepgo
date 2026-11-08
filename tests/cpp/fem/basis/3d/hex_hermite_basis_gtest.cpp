#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace litepgo::fem::test {

namespace {
static_assert(litepgo::fem::Basis3D<HexHermiteBasis>);
static_assert(HexHermiteBasis::num_basis_functions == 64);

Eigen::Matrix<double, 8, 3> corners() {
    Eigen::Matrix<double, 8, 3> result;
    result << 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1;
    return result;
}

const std::array<Eigen::Vector3d, 6> samples{{Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(1, 1, 1),
                                              Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(0, 0.3, 0.7),
                                              Eigen::Vector3d(0.2, 0.3, 0.4), Eigen::Vector3d(1.2, -0.1, 0.8)}};

// The specified vertex slots, independent of the implementation's tensor permutation.
const std::array<Eigen::Vector3i, 8> derivatives{
    {Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(0, 1, 0), Eigen::Vector3i(0, 0, 1),
     Eigen::Vector3i(1, 1, 0), Eigen::Vector3i(1, 0, 1), Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(1, 1, 1)}};

const std::array<Eigen::Vector3i, 6> second_derivatives{Eigen::Vector3i(2, 0, 0), Eigen::Vector3i(1, 1, 0),
                                                        Eigen::Vector3i(1, 0, 1), Eigen::Vector3i(0, 2, 0),
                                                        Eigen::Vector3i(0, 1, 1), Eigen::Vector3i(0, 0, 2)};

double monomial(const Eigen::Vector3i& powers, const Eigen::Vector3d& xi, const Eigen::Vector3i& derivative) {
    double result = 1;
    for (int d = 0; d < 3; ++d) {
        if (powers[d] < derivative[d])
            return 0;
        for (int order = 0; order < derivative[d]; ++order)
            result *= powers[d] - order;
        result *= std::pow(xi[d], powers[d] - derivative[d]);
    }
    return result;
}
} // namespace

TEST(HexHermiteBasis, InterpolatesCornerValuesAndFirstDerivativesInVertexOrder) {
    const auto nodes = corners();
    for (int vertex = 0; vertex < 8; ++vertex) {
        const auto N = HexHermiteBasis::eval_N(nodes.row(vertex).transpose());
        const auto dN_dxi = HexHermiteBasis::eval_dN_dxi(nodes.row(vertex).transpose());
        for (int slot = 0; slot < 8; ++slot)
            EXPECT_EQ(HexHermiteBasis::coefficient_info(8 * vertex + slot).point, nodes.row(vertex).transpose());
        EXPECT_EQ(N, (Eigen::Matrix<double, 64, 1>::Unit(8 * vertex)));
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(dN_dxi.col(d), (Eigen::Matrix<double, 64, 1>::Unit(8 * vertex + d + 1)));
    }
    EXPECT_THROW(HexHermiteBasis::coefficient_info(-1), std::out_of_range);
    EXPECT_THROW(HexHermiteBasis::coefficient_info(64), std::out_of_range);
}

TEST(HexHermiteBasis, ReproducesAllTricubicMonomialsAndGradients) {
    const auto nodes = corners();
    for (int i = 0; i <= 3; ++i)
        for (int j = 0; j <= 3; ++j)
            for (int k = 0; k <= 3; ++k) {
                const Eigen::Vector3i powers(i, j, k);
                SCOPED_TRACE(powers.transpose());
                Eigen::Matrix<double, 64, 1> coefficients;
                for (int vertex = 0; vertex < 8; ++vertex)
                    for (int slot = 0; slot < 8; ++slot) {
                        const auto info = HexHermiteBasis::coefficient_info(8 * vertex + slot);
                        EXPECT_EQ(info.point, nodes.row(vertex).transpose());
                        const Eigen::Vector3i order(info.derivative_order[0], info.derivative_order[1],
                                                    info.derivative_order[2]);
                        EXPECT_EQ(order, derivatives[static_cast<std::size_t>(slot)]);
                        coefficients[8 * vertex + slot] = monomial(powers, info.point, order);
                    }
                for (const auto& xi : samples) {
                    const auto N = HexHermiteBasis::eval_N(xi);
                    const auto dN_dxi = HexHermiteBasis::eval_dN_dxi(xi);
                    const auto d2N_dxi2 = HexHermiteBasis::eval_d2N_dxi2(xi);
                    EXPECT_NEAR(coefficients.dot(N), monomial(powers, xi, Eigen::Vector3i::Zero()), 1e-13);
                    for (int d = 0; d < 3; ++d)
                        EXPECT_NEAR(coefficients.dot(dN_dxi.col(d)), monomial(powers, xi, Eigen::Vector3i::Unit(d)),
                                    1e-13);
                    for (int s = 0; s < 6; ++s)
                        EXPECT_NEAR(coefficients.dot(d2N_dxi2.col(s)),
                                    monomial(powers, xi, second_derivatives[static_cast<std::size_t>(s)]), 3e-13);
                }
            }
}

TEST(HexHermiteBasis, ReproducesAffineVectorFieldsWith192Dofs) {
    const auto nodes = corners();
    Eigen::Matrix3d A;
    A << 0.2, -0.1, 0.3, 0.4, 0.2, -0.3, -0.2, 0.5, 0.1;
    const Eigen::Vector3d b(-0.3, 0.4, 0.7);
    Eigen::Matrix<double, 64, 3> coefficients = Eigen::Matrix<double, 64, 3>::Zero();
    for (int vertex = 0; vertex < 8; ++vertex) {
        coefficients.row(8 * vertex) = nodes.row(vertex) * A.transpose() + b.transpose();
        for (int d = 0; d < 3; ++d)
            coefficients.row(8 * vertex + d + 1) = A.col(d).transpose();
    }
    auto interpolation = std::make_shared<litepgo::fem::InterpolatedResponse<HexHermiteBasis>>(coefficients);
    EXPECT_EQ(interpolation->num_inputs(), 3);
    for (const auto& xi : samples) {
        EXPECT_TRUE(interpolation->eval_value(xi).isApprox(A * xi + b, 1e-13));
        EXPECT_TRUE(Eigen::MatrixXd(interpolation->eval_jacobian(xi)).isApprox(A, 1e-13));
        EXPECT_TRUE(Eigen::MatrixXd(interpolation->eval_hessian(xi, Vec::Ones(3))).isZero(1e-13));
    }
}

TEST(HexHermiteBasis, GradientsMatchFiniteDifferences) {
    constexpr double h = 1e-6;
    for (const auto& xi : samples) {
        const auto dN_dxi = HexHermiteBasis::eval_dN_dxi(xi);
        const auto d2N_dxi2 = HexHermiteBasis::eval_d2N_dxi2(xi);
        for (int d = 0; d < 3; ++d) {
            Eigen::Vector3d plus = xi, minus = xi;
            plus[d] += h;
            minus[d] -= h;
            const Eigen::Matrix<double, 64, 1> fd =
                (HexHermiteBasis::eval_N(plus) - HexHermiteBasis::eval_N(minus)) / (2 * h);
            EXPECT_TRUE((fd - dN_dxi.col(d)).isZero(2e-9));
            const Eigen::Matrix<double, 64, 3> second_fd =
                (HexHermiteBasis::eval_dN_dxi(plus) - HexHermiteBasis::eval_dN_dxi(minus)) / (2 * h);
            constexpr int columns[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
            for (int e = 0; e < 3; ++e)
                EXPECT_TRUE((second_fd.col(e) - d2N_dxi2.col(columns[d][e])).isZero(5e-9));
        }
    }
}

} // namespace litepgo::fem::test
