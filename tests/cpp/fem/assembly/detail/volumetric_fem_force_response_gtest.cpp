#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_energy.hpp"
#include "litepgo/fem/assembly/volumetric_fem_force_fp_response.hpp"
#include "litepgo/fem/assembly/volumetric_fem_force_material_response.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/pointwise_volumetric_adapter.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>

namespace litepgo::fem::test {
namespace {

template <class B, class Q> void check_hex_responses(bool warped = false) {
    const mesh::HexTopology topology(12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}});
    const dof::VertexDofMap map(topology, BasisDofTransform<B>::dofs_per_vertex);
    std::vector<std::shared_ptr<litepgo::energy::Response>> geometry;
    for (int e = 0; e < 2; ++e) {
        Eigen::MatrixXd X(8, 3);
        for (int v = 0; v < 8; ++v) {
            const auto xi = HexQ1Basis::coefficient_info(v).point;
            X.row(v) = Eigen::Vector3d(2.0 + 1.7 * (xi.x() + e), -.3 + .8 * xi.y(), .4 + 1.2 * xi.z());
            if (warped)
                X(v, 0) += .05 * xi.y() * xi.z();
        }
        geometry.push_back(std::make_shared<InterpolatedResponse<HexQ1Basis>>(X));
    }
    Vec u(map.num_dofs());
    for (int i = 0; i < u.size(); ++i)
        u[i] = .002 * std::sin(i + .3);
    const Eigen::Vector2d theta(2, 3);
    Eigen::Matrix3d Fp;
    Fp << 1.1, .05, 0, .02, .95, .03, 0, .04, 1.03;
    const Vec p = Fp.reshaped();
    const std::vector<std::shared_ptr<litepgo::energy::Response>> plastic(
        2, std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((p).size(), 3), p)),
        parameters(2, std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((theta).size(), 3), theta));
    using Energy = VolumetricFemEnergy<B, Q>;
    using MaterialForce = VolumetricFemForceMaterialResponse<B, Q>;
    using PlasticForce = VolumetricFemForceFpResponse<B, Q>;
    const Vec expected = Energy(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(), map, geometry,
                                plastic, parameters)
                             .eval_gradient(u);
    Eigen::MatrixXd material_reference, plastic_reference;
    for (int batch_size : {1, 3}) {
        MaterialForce material(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(),
                               dof::ConstantDofMap(topology, 2), map, geometry, u, plastic, batch_size);
        PlasticForce plastic_force(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(),
                                   dof::ConstantDofMap(topology, 9), map, geometry, u, parameters, batch_size);
        EXPECT_TRUE(material.eval_value(theta).isApprox(expected, 1e-11));
        EXPECT_TRUE(plastic_force.eval_value(p).isApprox(expected, 1e-11));
        const Eigen::MatrixXd M = material.eval_jacobian(theta), P = plastic_force.eval_jacobian(p);
        EXPECT_TRUE((M * theta).isApprox(expected, 1e-11));
        if (batch_size == 1) {
            material_reference = M;
            plastic_reference = P;
        } else {
            EXPECT_EQ((M - material_reference).norm(), 0);
            EXPECT_EQ((P - plastic_reference).norm(), 0);
        }
        const Vec direction = Vec::LinSpaced(9, -.2, .3);
        const double h = 1e-6;
        const std::vector<std::shared_ptr<litepgo::energy::Response>> plus(
            2, std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((p + h * direction).size(), 3),
                                                                 p + h * direction)),
            minus(2, std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((p - h * direction).size(), 3),
                                                                       p - h * direction));
        const Vec fd = (Energy(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(), map, geometry,
                               plus, parameters)
                            .eval_gradient(u) -
                        Energy(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(), map, geometry,
                               minus, parameters)
                            .eval_gradient(u)) /
                       (2 * h);
        EXPECT_TRUE((P * direction).isApprox(fd, 1e-8));
        MaterialForce material_elements(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(),
                                        dof::ElementDofMap(topology, 2), map, geometry, u, plastic, batch_size);
        PlasticForce plastic_elements(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(),
                                      dof::ElementDofMap(topology, 9), map, geometry, u, parameters, batch_size);
        Vec ts(4), ps(18);
        ts << theta, theta;
        ps << p, p;
        const Eigen::MatrixXd Me = material_elements.eval_jacobian(ts), Pe = plastic_elements.eval_jacobian(ps);
        EXPECT_TRUE(M.isApprox(Me.leftCols(2) + Me.rightCols(2), 1e-13));
        EXPECT_TRUE(P.isApprox(Pe.leftCols(9) + Pe.rightCols(9), 1e-13));
        EXPECT_TRUE(plastic_elements.eval_value(ps).isApprox(expected, 1e-11));
        // Unequal element parameters expose column swaps hidden by shared-block sums.
        ts.tail(2) *= 1.3;
        ps.tail(9) *= 1.05;
        const Eigen::MatrixXd heterogeneous_M = material_elements.eval_jacobian(ts);
        const Eigen::MatrixXd heterogeneous_P = plastic_elements.eval_jacobian(ps);
        const auto material_gradient = [&](const Vec& t) -> Vec {
            const std::vector<std::shared_ptr<litepgo::energy::Response>> fields{
                std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((t.head(2)).size(), 3), t.head(2)),
                std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((t.tail(2)).size(), 3), t.tail(2))};
            return Energy(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(), map, geometry,
                          plastic, fields)
                .eval_gradient(u);
        };
        const auto plastic_gradient = [&](const Vec& p_element) -> Vec {
            const std::vector<std::shared_ptr<litepgo::energy::Response>> fields{
                std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((p_element.head(9)).size(), 3),
                                                                  p_element.head(9)),
                std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat((p_element.tail(9)).size(), 3),
                                                                  p_element.tail(9))};
            return Energy(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(), map, geometry,
                          fields, parameters)
                .eval_gradient(u);
        };
        EXPECT_TRUE(material_elements.eval_value(ts).isApprox(material_gradient(ts), 1e-11));
        EXPECT_TRUE(plastic_elements.eval_value(ps).isApprox(plastic_gradient(ps), 1e-11));
        for (double step : {1e-4, 1e-6}) {
            SCOPED_TRACE(step);
            for (int c = 0; c < ts.size(); ++c) {
                SCOPED_TRACE(c);
                Vec forward = ts, backward = ts;
                forward[c] += step;
                backward[c] -= step;
                const Vec numerical = (material_gradient(forward) - material_gradient(backward)) / (2 * step);
                EXPECT_LE((heterogeneous_M.col(c) - numerical).lpNorm<Eigen::Infinity>(),
                          1e-7 * (1 + numerical.lpNorm<Eigen::Infinity>()));
            }
            for (int c = 0; c < ps.size(); ++c) {
                SCOPED_TRACE(c);
                Vec forward = ps, backward = ps;
                forward[c] += step;
                backward[c] -= step;
                const Vec numerical = (plastic_gradient(forward) - plastic_gradient(backward)) / (2 * step);
                EXPECT_LE((heterogeneous_P.col(c) - numerical).lpNorm<Eigen::Infinity>(),
                          1e-7 * (1 + numerical.lpNorm<Eigen::Infinity>()));
            }
        }
        EXPECT_THROW(plastic_force.eval_jacobian(Vec::Zero(9)), std::domain_error);
        EXPECT_TRUE(plastic_force.eval_value(p).isApprox(expected, 1e-11));
    }
    const dof::VertexDofMap wrong(topology, BasisDofTransform<B>::dofs_per_vertex == 3 ? 24 : 3);
    EXPECT_THROW(
        (MaterialForce(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(),
                       dof::ConstantDofMap(topology, 2), wrong, geometry, Vec::Zero(wrong.num_dofs()), plastic)),
        std::invalid_argument);
    EXPECT_THROW(
        (PlasticForce(std::make_shared<litepgo::fem::PointwiseVolumetricAdapter<NeoHookean>>(),
                      dof::ConstantDofMap(topology, 9), wrong, geometry, Vec::Zero(wrong.num_dofs()), parameters)),
        std::invalid_argument);
}

TEST(VolumetricFemForceResponse, HexQ1WarpedGeometryMatchesEnergy) {
    check_hex_responses<HexQ1Basis, GaussLegendre3D<2>>(true);
}
TEST(VolumetricFemForceResponse, HexHermitePhysicalDofsMatchEnergyOnNonunitBoxes) {
    check_hex_responses<HexHermiteBasis, GaussLegendre3D<2>>();
}

TEST(VolumetricFemForceResponse, HexQ1FourthOrderQuadratureParameterDerivatives) {
    check_hex_responses<HexQ1Basis, GaussLegendre3D<4>>(true);
}
TEST(VolumetricFemForceResponse, HexHermiteFourthOrderQuadratureParameterDerivatives) {
    check_hex_responses<HexHermiteBasis, GaussLegendre3D<4>>();
}

TEST(BasisDofTransformResponse, HermiteScalesOnlyRowsInPaddedViews) {
    Eigen::MatrixXd X(8, 3);
    for (int v = 0; v < 8; ++v)
        X.row(v) = HexQ1Basis::coefficient_info(v).point.cwiseProduct(Eigen::Vector3d(2, 3, 4));
    auto geometry = std::make_shared<InterpolatedResponse<HexQ1Basis>>(X);
    const BasisDofTransform<HexHermiteBasis> transform(*geometry);
    Eigen::MatrixXd storage = Eigen::MatrixXd::Constant(194, 5, 7);
    StridedVecRef value(storage.col(0).segment(1, 192));
    StridedMatRef J(storage.block(1, 1, 192, 3));
    value.setOnes();
    J.setOnes();
    transform.pullback_response_into(value, &J);
    const std::array<double, 8> scales{1, 2, 3, 4, 6, 8, 12, 24};
    for (int i = 0; i < 192; ++i) {
        EXPECT_EQ(value[i], scales[(i / 3) % 8]);
        EXPECT_TRUE(J.row(i).isConstant(value[i]));
    }
    EXPECT_TRUE(storage.row(0).isConstant(7));
    EXPECT_TRUE(storage.row(193).isConstant(7));
    EXPECT_TRUE(storage.col(4).isConstant(7));
    StridedMatRef bad(storage.block(1, 1, 191, 3));
    const Vec before = value;
    EXPECT_THROW(transform.pullback_response_into(value, &bad), std::invalid_argument);
    EXPECT_EQ(value, before);
}
} // namespace
} // namespace litepgo::fem::test
