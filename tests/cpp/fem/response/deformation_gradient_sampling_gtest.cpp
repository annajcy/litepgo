#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/response/deformation_gradient_response.hpp"
#include "litepgo/fem/response/deformation_gradient_sampling_response.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <memory>

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace litepgo::fem::test {
namespace {
template <class B, class XBasis, class Q> void check_against_spatial_response() {
    Eigen::MatrixXd positions(XBasis::num_basis_functions, 3);
    for (int i = 0; i < positions.rows(); ++i) {
        const auto xi = XBasis::coefficient_info(i).point;
        positions.row(i) = Eigen::Vector3d(2 * xi.x() + 0.1 * xi.y() * xi.z(), 3 * xi.y(), 4 * xi.z());
    }
    auto X = std::make_shared<InterpolatedResponse<XBasis>>(positions);
    // The cache owns its data, including when the source geometry has gone away.
    auto cache = [&] {
        auto temporary = std::make_shared<InterpolatedResponse<XBasis>>(positions);
        return deformation_gradient_sampling_response<B>(*temporary, Q::points());
    }();
    const Eigen::MatrixXd A = cache.eval_jacobian(Vec::Zero(3 * B::num_basis_functions));
    for (int pose : {0, 1, 2, 0}) {
        Eigen::MatrixXd u(B::num_basis_functions, 3);
        for (int i = 0; i < u.rows(); ++i)
            for (int c = 0; c < 3; ++c)
                u(i, c) = 0.01 * std::sin(pose + 3 * i + c);
        auto displacement = std::make_shared<InterpolatedResponse<B>>(u);
        auto field = std::make_shared<DeformationGradientResponse>(displacement, X);
        const Eigen::VectorXd coefficients = u.reshaped<Eigen::RowMajor>();
        const Vec sampled = cache.eval_value(coefficients);
        for (int q = 0; q < Q::num_points; ++q) {
            const Eigen::Vector3d xi = Q::points().row(q).transpose();
            EXPECT_TRUE(sampled.segment<9>(9 * q).isApprox(field->eval_value(xi), 1e-12));
            const double h = 1e-6;
            for (int j = 0; j < coefficients.size(); ++j) {
                Eigen::MatrixXd plus = u, minus = u;
                plus(j / 3, j % 3) += h;
                minus(j / 3, j % 3) -= h;
                const Eigen::Matrix3d J = X->eval_jacobian(xi);
                const Eigen::Matrix3d difference =
                    (plus - minus).transpose() * B::eval_dN_dxi(xi) * J.inverse() / (2 * h);
                EXPECT_TRUE(A.middleRows(9 * q, 9).col(j).isApprox(difference.reshaped(), 1e-8));
            }
        }
    }
}
} // namespace

TEST(DeformationGradientSampling, MatchesSpatialResponsesAcrossQuadraturePointsAndDisplacements) {
    check_against_spatial_response<TetP1Basis, TetP1Basis, TetCentroidQuadrature>();
    check_against_spatial_response<HexQ1Basis, HexQ1Basis, GaussLegendre3D<2>>();
    check_against_spatial_response<HexHermiteBasis, HexQ1Basis, GaussLegendre3D<4>>();
}

} // namespace litepgo::fem::test
