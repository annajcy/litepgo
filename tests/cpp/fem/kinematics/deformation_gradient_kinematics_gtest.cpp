#include "litepgo/fem/basis/3d/hex_hermite_basis.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/basis/3d/tet_p1_basis.hpp"
#include "litepgo/fem/kinematics/deformation_gradient_kinematics.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace litepgo::fem::test {
namespace {
template <class B, class XBasis, class Q> void check_against_field() {
    Eigen::MatrixXd positions(XBasis::num_basis_functions, 3);
    for (int i = 0; i < positions.rows(); ++i) {
        const auto xi = XBasis::reference_node(i);
        positions.row(i) = Eigen::Vector3d(2 * xi.x() + 0.1 * xi.y() * xi.z(), 3 * xi.y(), 4 * xi.z());
    }
    const InterpolatedField3D<XBasis> X(positions);
    // The cache owns its data, including when the source geometry has gone away.
    const auto cache = [&] {
        const InterpolatedField3D<XBasis> temporary(positions);
        return DeformationGradientKinematics3D<B, Q>(temporary);
    }();
    for (int pose : {0, 1, 2, 0}) {
        Eigen::MatrixXd u(B::num_basis_functions, 3);
        for (int i = 0; i < u.rows(); ++i)
            for (int c = 0; c < 3; ++c)
                u(i, c) = 0.01 * std::sin(pose + 3 * i + c);
        const InterpolatedField3D<B> displacement(u);
        const DeformationGradientField3D field(displacement, X);
        const Eigen::VectorXd coefficients = u.reshaped<Eigen::RowMajor>();
        for (int q = 0; q < Q::num_points; ++q) {
            const Eigen::Vector3d xi = Q::points().row(q).transpose();
            EXPECT_TRUE(cache.eval(q, coefficients).reshaped().isApprox(field.eval_value(xi), 1e-12));
            EXPECT_TRUE(cache.derivative(q).isApprox(field.eval_dvalue_dcoeffs(xi), 1e-12));
        }
    }
}
} // namespace

TEST(DeformationGradientKinematics, MatchesFieldsAcrossQuadraturePointsAndDisplacements) {
    check_against_field<TetP1Basis, TetP1Basis, TetCentroidQuadrature>();
    check_against_field<HexQ1Basis, HexQ1Basis, GaussLegendre3D<2>>();
    check_against_field<HexHermiteBasis, HexQ1Basis, GaussLegendre3D<4>>();
}

TEST(DeformationGradientKinematics, RejectsInvalidIndicesCoefficientsAndGeometry) {
    Eigen::MatrixXd positions(4, 3);
    for (int i = 0; i < 4; ++i)
        positions.row(i) = TetP1Basis::reference_node(i);
    const InterpolatedField3D<TetP1Basis> X(positions);
    using Kinematics = DeformationGradientKinematics3D<TetP1Basis, TetCentroidQuadrature>;
    const Kinematics cache(X);
    EXPECT_THROW(cache.derivative(-1), std::out_of_range);
    EXPECT_THROW(cache.derivative(1), std::out_of_range);
    EXPECT_THROW(cache.eval(0, Eigen::VectorXd::Zero(11)), std::invalid_argument);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(12);
    u[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(cache.eval(0, u), std::invalid_argument);
    const InterpolatedField3D<TetP1Basis> degenerate(Eigen::MatrixXd::Zero(4, 3));
    EXPECT_THROW((Kinematics(degenerate)), std::invalid_argument);
}
} // namespace litepgo::fem::test
