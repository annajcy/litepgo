#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/scalar/energy.hpp"
#include "litepgo/fem/basis/3d/hex_q1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "test_material_sources.hpp"

#include <Eigen/LU>
#include <stdexcept>

#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_local_energy.hpp"
#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace litepgo::fem::test {

namespace {

const auto element_energy = [](const Field3D& X_field, const LocalEnergy3D& energy, Eigen::VectorXd* g = nullptr,
                               Eigen::MatrixXd* H = nullptr) {
    if (g)
        g->setZero(energy.num_dofs());
    if (H)
        H->setZero(energy.num_dofs(), energy.num_dofs());
    using Q = GaussLegendre3D<2>;
    double result = 0;
    for (int q = 0; q < Q::num_points; ++q) {
        const Eigen::Vector3d xi = Q::points().row(q).transpose();
        const auto derivative = X_field.eval_dvalue_dxi(xi);
        if (derivative.rows() != 3 || derivative.cols() != 3 || !derivative.allFinite())
            throw std::invalid_argument("Invalid coordinate derivative in test fixture");
        const double determinant = Eigen::Matrix3d(derivative).determinant();
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::invalid_argument("Invalid coordinate determinant in test fixture");
        result += energy.accumulate(xi, Q::weights()[q] * determinant, g, H);
    }
    return result;
};
struct Fixture {
    mesh::HexTopology topology{12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}}};
    Eigen::Matrix<double, 12, 3> X;
    Vec u = Vec::Zero(36);
    Vec fp = Vec::Zero(18);
    Vec theta{Eigen::Vector2d(2, 3)};

    Fixture() {
        for (int e = 0; e < 2; ++e) {
            const auto& v = topology.cell_vertices(e);
            for (int i = 0; i < 8; ++i) {
                const Eigen::Vector3d p = HexQ1Basis::coefficient_info(i).point + Eigen::Vector3d(e, 0, 0);
                X.row(v[static_cast<std::size_t>(i)]) = Eigen::RowVector3d(
                    1.2 * p.x() + 0.1 * p.y() * p.z(), 0.9 * p.y() + 0.05 * p.x() * p.z(), 1.1 * p.z());
            }
            Eigen::Matrix3d p;
            p << 1.1 + 0.1 * e, 0.08, -0.03, 0.02, 0.9 + 0.05 * e, 0.04, -0.01, 0.03, 1.05;
            fp.segment<9>(9 * e) = p.reshaped();
        }
        for (int j = 0; j < u.size(); ++j)
            u[j] = 0.04 * std::sin(0.7 * (j + 1));
    }
    InterpolatedField3D<HexQ1Basis> X_field(int e) const {
        Eigen::Matrix<double, 8, 3> local;
        const auto& v = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i)
            local.row(i) = X.row(v[static_cast<std::size_t>(i)]);
        return InterpolatedField3D<HexQ1Basis>(local);
    }
    auto fields(int e) const {
        const dof::VertexDofMap map(topology, 3);
        return MaterialSources{
            OwnedDeformationField(InterpolatedField3D<litepgo::fem::HexQ1Basis>(map.gather(e, u)), X_field(e)),
            IdentityField3D(dof::ElementDofMap(topology, 9).gather(e, fp)),
            IdentityField3D(dof::ConstantDofMap(topology, 2).gather(e, theta))};
    }
};

template <int NumDofs, class MakeFields, class BindEnergy>
void check_fd(const Field3D& X_field, const Eigen::Matrix<double, NumDofs, 1>& x, MakeFields make_fields,
              BindEnergy bind_energy) {
    const auto fields = make_fields(x);
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    element_energy(X_field, bind_energy(fields), &g, &H);
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    for (double h : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(h);
        Eigen::Matrix<double, NumDofs, 1> dg(x.size());
        Eigen::Matrix<double, NumDofs, NumDofs> dH(x.size(), x.size());
        for (int j = 0; j < x.size(); ++j) {
            auto plus = x, minus = x;
            plus[j] += h;
            minus[j] -= h;
            const auto plus_fields = make_fields(plus), minus_fields = make_fields(minus);
            Eigen::VectorXd plus_g, minus_g;
            dg[j] = (element_energy(X_field, bind_energy(plus_fields), &plus_g) -
                     element_energy(X_field, bind_energy(minus_fields), &minus_g)) /
                    (2 * h);
            dH.col(j) = (plus_g - minus_g) / (2 * h);
        }
        EXPECT_LT((g - dg).norm() / (1 + g.norm()), 20 * h * h + 1e-9);
        EXPECT_LT((H - dH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
    }
    Eigen::Matrix<double, NumDofs, 1> d(x.size()), v(x.size());
    for (int j = 0; j < x.size(); ++j) {
        d[j] = std::sin(j + 0.5);
        v[j] = std::cos(j + 0.2);
    }
    d.normalize();
    v.normalize();
    const double h = 3e-4;
    const double fd = (element_energy(X_field, bind_energy(make_fields(x + h * d + h * v))) -
                       element_energy(X_field, bind_energy(make_fields(x + h * d - h * v))) -
                       element_energy(X_field, bind_energy(make_fields(x - h * d + h * v))) +
                       element_energy(X_field, bind_energy(make_fields(x - h * d - h * v)))) /
                      (4 * h * h);
    EXPECT_NEAR(fd, d.dot(H * v), 2e-6 * (1 + H.norm()));
}

TEST(MaterialSources, MixedRepresentationsPointInputsAndLocalFiniteDifferences) {
    const Fixture f;
    const dof::VertexDofMap map(f.topology, 3);
    for (int e = 0; e < 2; ++e) {
        SCOPED_TRACE(e);
        const auto X_field = f.X_field(e);
        const auto fields = f.fields(e);
        const auto ue = map.gather(e, f.u);
        for (int q = 0; q < GaussLegendre3D<2>::num_points; ++q) {
            const Eigen::Vector3d ref_xi = GaussLegendre3D<2>::points().row(q).transpose();
            const auto point = fields.eval(ref_xi);
            EXPECT_TRUE(point.F.isApprox(Eigen::Matrix3d::Identity() +
                                         InterpolatedField3D<HexQ1Basis>(ue).eval_dvalue_dxi(ref_xi) *
                                             X_field.eval_dvalue_dxi(ref_xi).inverse()));
            EXPECT_EQ(point.Fp, f.fp.segment<9>(9 * e).reshaped(3, 3));
            EXPECT_EQ(point.theta, dof::ConstantDofMap(f.topology, 2).gather(0, f.theta));
            EXPECT_TRUE(fields.Fp.eval_dvalue_dcoeffs(ref_xi).isIdentity());
            EXPECT_TRUE(fields.theta.eval_dvalue_dcoeffs(ref_xi).isIdentity());
        }
        const Eigen::Matrix<double, 24, 1> uvec = ue.reshaped<Eigen::RowMajor>();
        check_fd(
            X_field, uvec,
            [&](const Eigen::Matrix<double, 24, 1>& x) {
                Eigen::Matrix<double, 8, 3> c;
                for (int j = 0; j < 24; ++j)
                    c(j / 3, j % 3) = x[j];
                return MaterialSources{OwnedDeformationField(InterpolatedField3D<litepgo::fem::HexQ1Basis>(c), X_field),
                                       IdentityField3D(dof::ElementDofMap(f.topology, 9).gather(e, f.fp)),
                                       IdentityField3D(dof::ConstantDofMap(f.topology, 2).gather(e, f.theta))};
            },
            [&](const auto& local_fields) {
                return bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(local_fields);
            });
    }
}

TEST(MaterialSources, SameQuadratureAcceptsDifferentActiveDofCounts) {
    Fixture fixture;
    fixture.u.setZero();
    const auto X_field = fixture.X_field(0);
    const auto nodal = fixture.fields(0);
    const MaterialSources direct{IdentityField3D(Eigen::Matrix3d::Identity().reshaped()), nodal.Fp, nodal.theta};
    Eigen::VectorXd nodal_gradient, direct_gradient;
    Eigen::MatrixXd nodal_hessian, direct_hessian;
    const double nodal_value = element_energy(X_field, bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(nodal),
                                              &nodal_gradient, &nodal_hessian);
    const double direct_value = element_energy(
        X_field, bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(direct), &direct_gradient, &direct_hessian);
    EXPECT_EQ(nodal_gradient.size(), 24);
    EXPECT_EQ(direct_gradient.size(), 9);
    EXPECT_DOUBLE_EQ(nodal_value, direct_value);
    Eigen::Matrix<double, 24, 1> expected_gradient = Eigen::Matrix<double, 24, 1>::Zero();
    Eigen::Matrix<double, 24, 24> expected_hessian = Eigen::Matrix<double, 24, 24>::Zero();
    Eigen::Matrix<double, 9, 9> expected_direct_hessian = Eigen::Matrix<double, 9, 9>::Zero();
    for (int q = 0; q < GaussLegendre3D<2>::num_points; ++q) {
        const Eigen::Vector3d point_xi = GaussLegendre3D<2>::points().row(q).transpose();
        const double weight = GaussLegendre3D<2>::weights()[q] * X_field.eval_dvalue_dxi(point_xi).determinant();
        const auto inputs = nodal.eval(point_xi);
        const auto dF_du_local = nodal.F.eval_dvalue_dcoeffs(point_xi);
        const auto stress = NeoHookean::eval_dW_dF(inputs);
        const auto tangent = NeoHookean::eval_d2W_dF2(inputs);
        expected_gradient +=
            weight * dF_du_local.transpose() * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(stress.data());
        expected_hessian += weight * dF_du_local.transpose() * tangent * dF_du_local;
        expected_direct_hessian += weight * tangent;
    }
    EXPECT_TRUE(nodal_gradient.isApprox(expected_gradient, 1e-13));
    EXPECT_TRUE(nodal_hessian.isApprox(expected_hessian, 1e-13));
    EXPECT_TRUE(direct_hessian.isApprox(expected_direct_hessian, 1e-13));
    const Eigen::Matrix<double, 9, 1> identity =
        Eigen::Map<const Eigen::Matrix<double, 9, 1>>(Eigen::Matrix3d::Identity().eval().data());
    check_fd<9>(
        X_field, identity,
        [&](const Eigen::Matrix<double, 9, 1>& x)
            -> MaterialSources<IdentityField3D, IdentityField3D, IdentityField3D> {
            return {IdentityField3D(x), nodal.Fp, nodal.theta};
        },
        [&](const auto& fields) { return bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(fields); });
}

TEST(MaterialSources, DisplacementDofNumbersAccumulateAndFieldsOwnSnapshots) {
    Fixture f;
    const dof::VertexDofMap map(f.topology, 3);
    const auto total = [&](const Fixture& data) {
        double value = 0;
        for (int e = 0; e < 2; ++e) {
            const auto X_field = data.X_field(e);
            value += element_energy(X_field, bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(data.fields(e)));
        }
        return value;
    };
    Vec gu = Vec::Zero(36);
    for (int e = 0; e < 2; ++e) {
        const auto X_field = f.X_field(e);
        const auto fields = f.fields(e);
        const auto indices = map.dof_indices(e);
        Eigen::VectorXd local_gradient;
        element_energy(X_field, bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(fields), &local_gradient);
        for (int j = 0; j < 24; ++j)
            gu[indices[j]] += local_gradient[j];
    }
    const double h = 1e-4;
    for (int j = 0; j < 36; ++j) {
        auto plus = f, minus = f;
        plus.u[j] += h;
        minus.u[j] -= h;
        EXPECT_NEAR(gu[j], (total(plus) - total(minus)) / (2 * h), 1e-7 * (1 + gu.norm()));
    }
    const auto before = f.fields(0);
    const Vec original = f.fp.head<9>();
    f.fp.head<9>() = 1.1 * original;
    f.theta << 4, 5;
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_EQ(before.Fp.eval_value(xi), original);
    EXPECT_EQ(before.theta.eval_value(xi), Eigen::Vector2d(2, 3));
    EXPECT_NE(f.fields(0).Fp.eval_value(xi), original);
    EXPECT_EQ(f.fields(1).theta.eval_value(xi), Eigen::Vector2d(4, 5));
}

TEST(MaterialSources, DirectInfeasiblePlasticStateReachesMaterialValidation) {
    const Fixture f;
    const auto X_field = f.X_field(0);
    auto fields = f.fields(0);
    Eigen::Matrix3d inverted = Eigen::Matrix3d::Identity();
    inverted(0, 0) = -1;
    fields.Fp = IdentityField3D(inverted.reshaped());
    Eigen::VectorXd g;
    Eigen::MatrixXd h;
    EXPECT_THROW(element_energy(X_field, bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(fields)),
                 std::domain_error);
    EXPECT_THROW(element_energy(X_field, bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(fields), &g),
                 std::domain_error);
    EXPECT_THROW(
        element_energy(X_field, bind_energy<VolumetricMaterialLocalEnergyF3D<NeoHookean>>(fields), nullptr, &h),
        std::domain_error);
}
} // namespace

} // namespace litepgo::fem::test
