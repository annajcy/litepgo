#include "litepgo/energy/response/affine_response.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/tet_centroid_quadrature.hpp"
#include "litepgo/fem/response/interpolated_response.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/fem/volumetric_fem_force_fp_response.hpp"
#include "litepgo/fem/volumetric_fem_force_material_response.hpp"
#include <gtest/gtest.h>
#include <memory>

namespace litepgo::fem::test {
namespace {

using Force = VolumetricFemForceMaterialResponse<NeoHookean, TetP1Basis, TetCentroidQuadrature>;
using Energy = VolumetricFemEnergy<NeoHookean, TetP1Basis, TetCentroidQuadrature>;
struct Fixture {
    mesh::TetTopology topology{5, {{0, 1, 2, 3}, {1, 2, 3, 4}}};
    dof::VertexDofMap map{topology, 3};
    std::vector<std::shared_ptr<litepgo::energy::Response>> X;
    std::vector<std::shared_ptr<litepgo::energy::Response>> plastic;
    Vec u;
    Fixture() {
        Eigen::Matrix<double, 5, 3, Eigen::RowMajor> nodes;
        nodes << 0, 0, 0, 1.2, 0, 0, 0, 1, 0, 0, 0, .8, 1.2, 1, .8;
        Eigen::Matrix3d F;
        F << 1.15, .12, 0, .03, .94, .08, 0, .02, 1.07;
        const Eigen::MatrixXd displacement = nodes * (F - Eigen::Matrix3d::Identity()).transpose();
        u = displacement.reshaped<Eigen::RowMajor>();
        const Vec positions = nodes.reshaped<Eigen::RowMajor>();
        for (int e = 0; e < 2; ++e) {
            X.push_back(std::make_shared<InterpolatedResponse<TetP1Basis>>(map.gather(e, positions)));
            Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity();
            Fp(0, 0) = 1.1 + .05 * e;
            Fp(1, 2) = .1;
            plastic.push_back(std::make_shared<litepgo::energy::AffineResponse>(
                litepgo::SpMat((Fp.reshaped()).size(), 3), Fp.reshaped()));
        }
    }
    Vec force(const Vec& theta) const {
        std::vector<std::shared_ptr<litepgo::energy::Response>> fields;
        for (int e = 0; e < 2; ++e)
            fields.push_back(std::make_shared<litepgo::energy::AffineResponse>(
                litepgo::SpMat((theta.segment(theta.size() == 2 ? 0 : 2 * e, 2)).size(), 3),
                theta.segment(theta.size() == 2 ? 0 : 2 * e, 2)));
        return Energy(map, X, plastic, fields).eval_gradient(u);
    }
};
TEST(VolumetricFemForceMaterialResponse, MatchesEnergySharedAndPerElementAndBatchSizes) {
    Fixture f;
    const Vec theta = Eigen::Vector2d(2, 3);
    const Vec independent = (Vec(4) << 2, 3, 5, .8).finished();
    Eigen::MatrixXd reference;
    for (int size : {1, 3, 256}) {
        Force shared(dof::ConstantDofMap(f.topology, 2), f.map, f.X, f.u, f.plastic, size);
        Force per_element(dof::ElementDofMap(f.topology, 2), f.map, f.X, f.u, f.plastic, size);
        EXPECT_EQ(shared.num_inputs(), 2);
        EXPECT_EQ(shared.num_outputs(), 15);
        EXPECT_TRUE(shared.eval_value(theta).isApprox(f.force(theta), 1e-12));
        EXPECT_TRUE(per_element.eval_value(independent).isApprox(f.force(independent), 1e-12));
        const Eigen::MatrixXd J = shared.eval_jacobian(theta), E = per_element.eval_jacobian(independent);
        EXPECT_TRUE(J.isApprox(E.leftCols(2) + E.rightCols(2), 1e-13));
        EXPECT_TRUE((E * independent).isApprox(f.force(independent), 1e-12));
        if (size == 1)
            reference = J;
        else
            EXPECT_EQ((reference - J).norm(), 0);
        const SpMat before = shared.eval_jacobian(theta);
        EXPECT_THROW(shared.eval_value(Eigen::Vector2d(2, 0)), std::invalid_argument);
        EXPECT_TRUE(shared.eval_value(theta).isApprox(f.force(theta), 1e-12));
        EXPECT_EQ(before.nonZeros(), shared.eval_jacobian(theta).nonZeros());
    }
}
TEST(VolumetricFemForceMaterialResponse, OwnsDisplacementAndValidatesConstruction) {
    Fixture f;
    const dof::ConstantDofMap p(f.topology, 2);
    const Vec theta = Eigen::Vector2d(2, 3), expected = f.force(theta);
    Force r(p, f.map, f.X, f.u, f.plastic);
    f.u.setZero();
    f.X.clear();
    f.plastic.clear();
    EXPECT_TRUE(r.eval_value(theta).isApprox(expected, 1e-12));
    Fixture good;
    EXPECT_THROW((Force(p, good.map, good.X, Vec::Zero(1), good.plastic)), std::invalid_argument);
    EXPECT_THROW((Force(p, good.map, {}, good.u, good.plastic)), std::invalid_argument);
    EXPECT_THROW((Force(p, good.map, good.X, good.u, {})), std::invalid_argument);
    EXPECT_THROW((Force(p, good.map, good.X, good.u, good.plastic, 0)), std::invalid_argument);
    EXPECT_THROW((Force(dof::ConstantDofMap(good.topology, 3), good.map, good.X, good.u, good.plastic)),
                 std::invalid_argument);
    const mesh::TetTopology empty(0, {});
    EXPECT_THROW((Force(dof::ConstantDofMap(empty, 2), good.map, good.X, good.u, good.plastic)), std::invalid_argument);
    Force zero(dof::ConstantDofMap(empty, 2), dof::VertexDofMap(empty, 3), {}, Vec(0), {});
    EXPECT_EQ(zero.eval_value(theta).size(), 0);
    EXPECT_EQ(zero.eval_jacobian(theta).nonZeros(), 0);
}
} // namespace
} // namespace litepgo::fem::test

namespace litepgo::fem::test {
TEST(VolumetricFemForceFpResponse, EnergyConsistencySharedElementAndBatchJacobians) {
    using PlasticForce = VolumetricFemForceFpResponse<NeoHookean, TetP1Basis, TetCentroidQuadrature>;
    Fixture f;
    std::vector<std::shared_ptr<litepgo::energy::Response>> theta{
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(2, 3)),
        std::make_shared<litepgo::energy::AffineResponse>(litepgo::SpMat(2, 3), Eigen::Vector2d(4, 2))};
    Vec p(18);
    for (int e = 0; e < 2; ++e)
        p.segment<9>(9 * e) = f.plastic[e]->eval_value(Eigen::Vector3d::Zero());
    Eigen::MatrixXd reference;
    for (int size : {1, 3, 256}) {
        PlasticForce local(dof::ElementDofMap(f.topology, 9), f.map, f.X, f.u, theta, size);
        PlasticForce shared(dof::ConstantDofMap(f.topology, 9), f.map, f.X, f.u, theta, size);
        const auto expected = Energy(f.map, f.X, f.plastic, theta).eval_gradient(f.u);
        EXPECT_TRUE(local.eval_value(p).isApprox(expected, 1e-12));
        const Eigen::MatrixXd J = local.eval_jacobian(p);
        if (size == 1)
            reference = J;
        else
            EXPECT_EQ((reference - J).norm(), 0);
        for (int c = 0; c < 18; ++c) {
            Vec plus = p, minus = p;
            plus[c] += 1e-6;
            minus[c] -= 1e-6;
            const Vec difference = (local.eval_value(plus) - local.eval_value(minus)) / 2e-6;
            EXPECT_TRUE(J.col(c).isApprox(difference, 1e-8));
        }
        Vec repeated(18);
        repeated << p.head(9), p.head(9);
        const Eigen::MatrixXd element_J = local.eval_jacobian(repeated);
        EXPECT_TRUE(Eigen::MatrixXd(shared.eval_jacobian(p.head(9)))
                        .isApprox(element_J.leftCols(9) + element_J.rightCols(9), 1e-13));
        EXPECT_TRUE(shared.eval_value(p.head(9)).isApprox(local.eval_value(repeated), 1e-13));
        Vec bad = p;
        bad.head(9).setZero();
        EXPECT_THROW(local.eval_jacobian(bad), std::domain_error);
        EXPECT_TRUE(local.eval_value(p).isApprox(expected, 1e-12));
        f.u.setZero();
        EXPECT_TRUE(local.eval_value(p).isApprox(expected, 1e-12));
        f = Fixture{};
    }
    EXPECT_THROW((PlasticForce(dof::ConstantDofMap(f.topology, 2), f.map, f.X, f.u, theta)), std::invalid_argument);
    EXPECT_THROW((PlasticForce(dof::ConstantDofMap(f.topology, 9), f.map, f.X, f.u, {})), std::invalid_argument);
    const mesh::TetTopology empty(0, {});
    PlasticForce zero(dof::ConstantDofMap(empty, 9), dof::VertexDofMap(empty, 3), {}, Vec(0), {});
    EXPECT_EQ(zero.eval_value(p.head(9)).size(), 0);
    EXPECT_EQ(zero.eval_jacobian(p.head(9)).cols(), 9);
}
} // namespace litepgo::fem::test
