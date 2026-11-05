#include "litepgo/fem/field/shared_field.hpp"

#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_field.hpp"
#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/volumetric_fem_energy.hpp"
#include "litepgo/mesh/mesh_topology.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace litepgo::fem::test {

namespace {

Eigen::MatrixXd unit_box_nodes() {
    Eigen::MatrixXd nodes(8, 3);
    for (int v = 0; v < 8; ++v)
        nodes.row(v) = HexQ1Basis::coefficient_info(v).point.transpose();
    return nodes;
}

} // namespace

TEST(SharedField, ForwardsEveryEvaluation) {
    Eigen::MatrixXd coeffs = unit_box_nodes();
    coeffs.col(0) *= 2;
    auto field = std::make_shared<const InterpolatedField3D<HexQ1Basis>>(coeffs);
    const SharedField3D shared(field);
    const Eigen::Vector3d xi(0.2, 0.7, 0.4);
    EXPECT_EQ(shared.num_components(), field->num_components());
    EXPECT_EQ(shared.num_dofs(), field->num_dofs());
    EXPECT_EQ(shared.eval_value(xi), field->eval_value(xi));
    EXPECT_EQ(shared.eval_dvalue_dcoeffs(xi), field->eval_dvalue_dcoeffs(xi));
    EXPECT_EQ(shared.eval_dvalue_dxi(xi), field->eval_dvalue_dxi(xi));
    EXPECT_EQ(shared.eval_dgrad_xi_dcoeffs(xi), field->eval_dgrad_xi_dcoeffs(xi));
    EXPECT_EQ(shared.eval_d2value_dxi2(xi), field->eval_d2value_dxi2(xi));
    EXPECT_EQ(shared.eval_dhess_xi_dcoeffs(xi), field->eval_dhess_xi_dcoeffs(xi));
}

TEST(SharedField, KeepsTheFieldAliveAndRejectsNull) {
    std::weak_ptr<const Field3D> weak;
    std::unique_ptr<SharedField3D> shared;
    {
        auto field = std::make_shared<const IdentityField3D>(Eigen::Vector2d(1, 2));
        weak = field;
        shared = std::make_unique<SharedField3D>(field);
    }
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(shared->eval_value(Eigen::Vector3d::Zero()), Eigen::Vector2d(1, 2));
    const SharedField3D copy = *shared;
    shared.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_THROW(SharedField3D(nullptr), std::invalid_argument);
}

TEST(SharedField, ErasedFieldsGiveTheSameVolumetricEnergy) {
    const mesh::HexTopology topology(8, {{0, 1, 2, 3, 4, 5, 6, 7}});
    const dof::VertexDofMap map(topology, 3);
    const Eigen::MatrixXd X = 1.5 * unit_box_nodes();
    const Eigen::Matrix3d Fp = Eigen::Matrix3d::Identity() + 0.1 * Eigen::Matrix3d::Ones();
    const Eigen::Vector2d theta(2, 3);
    using XField = InterpolatedField3D<HexQ1Basis>;
    const VolumetricFemEnergy<NeoHookean, HexQ1Basis, GaussLegendre3D<2>, XField, IdentityField3D, IdentityField3D>
        concrete(map, {XField(X)}, {IdentityField3D(Fp.reshaped())}, {IdentityField3D(theta)});
    const VolumetricFemEnergy<NeoHookean, HexQ1Basis, GaussLegendre3D<2>, SharedField3D, SharedField3D, SharedField3D>
        erased(map, {SharedField3D(std::make_shared<const XField>(X))},
               {SharedField3D(std::make_shared<const IdentityField3D>(Fp.reshaped()))},
               {SharedField3D(std::make_shared<const IdentityField3D>(theta))});
    const Vec u = 0.05 * Vec::LinSpaced(24, -1, 1);
    EXPECT_EQ(erased.eval_value(u), concrete.eval_value(u));
    EXPECT_EQ(erased.eval_gradient(u), concrete.eval_gradient(u));
    EXPECT_EQ(Eigen::MatrixXd(erased.eval_hessian(u)), Eigen::MatrixXd(concrete.eval_hessian(u)));
}

} // namespace litepgo::fem::test
