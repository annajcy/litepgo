#include "../local/quadratic_parameter_material.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "test_material_sources.hpp"

#include "litepgo/fem/local/integrate_volume.hpp"
#include "litepgo/fem/local/material_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/tensor_product_quadrature.hpp"

#include "litepgo/fem/kinematics/reference_mapping.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {
using namespace litepgo;
using namespace litepgo::test;
struct Fixture {
    HexTopology topology{12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}}};
    Eigen::Matrix<double, 12, 3> X;
    Vec u = Vec::Zero(36);
    Vec fp = Vec::Zero(18);
    Vec theta{Eigen::Vector2d(2, 3)};

    Fixture() {
        for (int e = 0; e < 2; ++e) {
            const auto& v = topology.cell_vertices(e);
            for (int i = 0; i < 8; ++i) {
                const Eigen::Vector3d p = HexQ1Basis::reference_node(i) + Eigen::Vector3d(e, 0, 0);
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
    InterpolatedField<HexQ1Basis> X_field(int e) const {
        Eigen::Matrix<double, 8, 3> local;
        const auto& v = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i)
            local.row(i) = X.row(v[static_cast<std::size_t>(i)]);
        return InterpolatedField<HexQ1Basis>(local);
    }
    auto fields(int e) const {
        const VertexDofMap map(topology, 3);
        return MaterialSources{
            DeformationGradient(InterpolatedField<litepgo::HexQ1Basis>(map.gather(e, u)), X_field(e)),
            IdentityField<Eigen::Matrix3d>(ElementDofMap(topology, 9).gather(e, fp).reshaped(3, 3)),
            IdentityField<Eigen::VectorXd>(ConstantDofMap(topology, 2).gather(e, theta))};
    }
};

template <int NumDofs, class MakeFields, class BindEnergy>
void check_fd(const InterpolatedField<HexQ1Basis>&, const Eigen::Matrix<double, NumDofs, 1>& x, MakeFields make_fields,
              BindEnergy bind_energy) {
    const auto fields = make_fields(x);
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    integrate_volume<GaussLegendre3D<2>>(bind_energy(fields), &g, &H);
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
            dg[j] = (integrate_volume<GaussLegendre3D<2>>(bind_energy(plus_fields), &plus_g) -
                     integrate_volume<GaussLegendre3D<2>>(bind_energy(minus_fields), &minus_g)) /
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
    const double fd = (integrate_volume<GaussLegendre3D<2>>(bind_energy(make_fields(x + h * d + h * v))) -
                       integrate_volume<GaussLegendre3D<2>>(bind_energy(make_fields(x + h * d - h * v))) -
                       integrate_volume<GaussLegendre3D<2>>(bind_energy(make_fields(x - h * d + h * v))) +
                       integrate_volume<GaussLegendre3D<2>>(bind_energy(make_fields(x - h * d - h * v)))) /
                      (4 * h * h);
    EXPECT_NEAR(fd, d.dot(H * v), 2e-6 * (1 + H.norm()));
}

TEST(MaterialSources, MixedRepresentationsPointInputsAndLocalFiniteDifferences) {
    const Fixture f;
    const VertexDofMap map(f.topology, 3);
    for (int e = 0; e < 2; ++e) {
        SCOPED_TRACE(e);
        const auto X_field = f.X_field(e);
        const auto fields = f.fields(e);
        const auto ue = map.gather(e, f.u);
        for (int q = 0; q < GaussLegendre3D<2>::num_points; ++q) {
            const auto ref =
                litepgo::ReferenceMapping(X_field.evaluate(GaussLegendre3D<2>::points().row(q).transpose()));
            const auto point = fields.evaluate(ref.xi());
            EXPECT_TRUE(point.F.isApprox(Eigen::Matrix3d::Identity() +
                                         InterpolatedField<HexQ1Basis>(ue).dvalue_dxi(ref.xi()) * ref.dxi_dX()));
            EXPECT_EQ(point.Fp, f.fp.segment<9>(9 * e).reshaped(3, 3));
            EXPECT_EQ(point.theta, ConstantDofMap(f.topology, 2).gather(0, f.theta));
            EXPECT_TRUE(fields.Fp.evaluate(ref.xi()).dvalue_dcoeffs().isIdentity());
            EXPECT_TRUE(fields.theta.evaluate(ref.xi()).dvalue_dcoeffs().isIdentity());
        }
        EXPECT_DOUBLE_EQ(
            integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::F>(X_field, fields)),
            integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Fp>(X_field, fields)));
        EXPECT_DOUBLE_EQ(
            integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::F>(X_field, fields)),
            integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Theta>(X_field, fields)));
        const Eigen::Matrix<double, 24, 1> uvec = ue.reshaped<Eigen::RowMajor>();
        check_fd(
            X_field, uvec,
            [&](const Eigen::Matrix<double, 24, 1>& x) {
                Eigen::Matrix<double, 8, 3> c;
                for (int j = 0; j < 24; ++j)
                    c(j / 3, j % 3) = x[j];
                return MaterialSources{
                    DeformationGradient(InterpolatedField<litepgo::HexQ1Basis>(c), X_field),
                    IdentityField<Eigen::Matrix3d>(ElementDofMap(f.topology, 9).gather(e, f.fp).reshaped(3, 3)),
                    IdentityField<Eigen::VectorXd>(ConstantDofMap(f.topology, 2).gather(e, f.theta))};
            },
            [&](const auto& local_fields) {
                return bind_energy<NeoHookean, MaterialVariable::F>(X_field, local_fields);
            });
        const Eigen::Matrix3d fp = f.fp.segment<9>(9 * e).reshaped(3, 3);
        const Eigen::Matrix<double, 9, 1> pvec = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fp.data());
        check_fd(
            X_field, pvec,
            [&](const Eigen::Matrix<double, 9, 1>& x) {
                return MaterialSources{
                    DeformationGradient(InterpolatedField<litepgo::HexQ1Basis>(ue), X_field),
                    IdentityField<Eigen::Matrix3d>(Eigen::Map<const Eigen::Matrix3d>(x.data())),
                    IdentityField<Eigen::VectorXd>(ConstantDofMap(f.topology, 2).gather(e, f.theta))};
            },
            [&](const auto& local_fields) {
                return bind_energy<NeoHookean, MaterialVariable::Fp>(X_field, local_fields);
            });
        check_fd(
            X_field, ConstantDofMap(f.topology, 2).gather(0, f.theta),
            [&](const Eigen::Matrix<double, 2, 1>& x) {
                return MaterialSources{DeformationGradient(InterpolatedField<litepgo::HexQ1Basis>(ue), X_field),
                                       IdentityField<Eigen::Matrix3d>(fp), IdentityField<Eigen::VectorXd>(x)};
            },
            [&](const auto& local_fields) {
                return bind_energy<NeoHookean, MaterialVariable::Theta>(X_field, local_fields);
            });
    }
}

TEST(MaterialSources, SameQuadratureAcceptsDifferentActiveDofCounts) {
    Fixture fixture;
    fixture.u.setZero();
    const auto X_field = fixture.X_field(0);
    const auto nodal = fixture.fields(0);
    const MaterialSources direct{IdentityField<Eigen::Matrix3d>(Eigen::Matrix3d::Identity()), nodal.Fp, nodal.theta};
    Eigen::VectorXd nodal_gradient, direct_gradient;
    Eigen::MatrixXd nodal_hessian, direct_hessian;
    const double nodal_value = integrate_volume<GaussLegendre3D<2>>(
        bind_energy<NeoHookean, MaterialVariable::F>(X_field, nodal), &nodal_gradient, &nodal_hessian);
    const double direct_value = integrate_volume<GaussLegendre3D<2>>(
        bind_energy<NeoHookean, MaterialVariable::F>(X_field, direct), &direct_gradient, &direct_hessian);
    EXPECT_EQ(nodal_gradient.size(), 24);
    EXPECT_EQ(direct_gradient.size(), 9);
    EXPECT_DOUBLE_EQ(nodal_value, direct_value);
    Eigen::Matrix<double, 24, 1> expected_gradient = Eigen::Matrix<double, 24, 1>::Zero();
    Eigen::Matrix<double, 24, 24> expected_hessian = Eigen::Matrix<double, 24, 24>::Zero();
    Eigen::Matrix<double, 9, 9> expected_direct_hessian = Eigen::Matrix<double, 9, 9>::Zero();
    for (int q = 0; q < GaussLegendre3D<2>::num_points; ++q) {
        const auto point = litepgo::ReferenceMapping(X_field.evaluate(GaussLegendre3D<2>::points().row(q).transpose()));
        const double weight = GaussLegendre3D<2>::weights()[q] * point.det_dX_dxi();
        const auto inputs = nodal.evaluate(point.xi());
        const auto dF_du_local = nodal.F.evaluate(point.xi()).dF_du_local();
        const auto stress = NeoHookean::dW_dF(inputs);
        const auto tangent = NeoHookean::d2W_dF2(inputs);
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
    check_fd(
        X_field, identity,
        [&](const Eigen::Matrix<double, 9, 1>& x) {
            return MaterialSources{IdentityField<Eigen::Matrix3d>(Eigen::Map<const Eigen::Matrix3d>(x.data())),
                                   nodal.Fp, nodal.theta};
        },
        [&](const auto& fields) { return bind_energy<NeoHookean, MaterialVariable::F>(X_field, fields); });
}

TEST(MaterialSources, ElementAndSharedDofNumbersAccumulateIntoDifferentGlobalBlocks) {
    Fixture f;
    const VertexDofMap map(f.topology, 3);
    std::array X_fields{f.X_field(0), f.X_field(1)};
    const auto total = [&](const Fixture& data) {
        double value = 0;
        for (int e = 0; e < 2; ++e)
            value += integrate_volume<GaussLegendre3D<2>>(
                bind_energy<NeoHookean, MaterialVariable::F>(X_fields[static_cast<std::size_t>(e)], data.fields(e)));
        return value;
    };
    Vec gu = Vec::Zero(36), gp = Vec::Zero(18), gt = Vec::Zero(2);
    Eigen::MatrixXd Hp = Eigen::MatrixXd::Zero(18, 18);
    Eigen::Matrix2d Ht = Eigen::Matrix2d::Zero();
    for (int e = 0; e < 2; ++e) {
        const auto& X_field = X_fields[static_cast<std::size_t>(e)];
        const auto fields = f.fields(e);
        const auto uidx = map.dof_indices(e);
        const auto pidx = ElementDofMap(f.topology, 9).dof_indices(e);
        const auto tidx = ConstantDofMap(f.topology, 2).dof_indices(e);
        Eigen::VectorXd ug, pg, tg;
        Eigen::MatrixXd ph, th;
        integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::F>(X_field, fields), &ug);
        integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Fp>(X_field, fields), &pg, &ph);
        integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Theta>(X_field, fields), &tg,
                                             &th);
        for (int j = 0; j < 24; ++j)
            gu[uidx[static_cast<std::size_t>(j)]] += ug[j];
        for (int j = 0; j < 9; ++j) {
            const int gj = pidx[static_cast<std::size_t>(j)];
            EXPECT_EQ(gj, 9 * e + j);
            gp[gj] += pg[j];
            for (int i = 0; i < 9; ++i)
                Hp(pidx[static_cast<std::size_t>(i)], gj) += ph(i, j);
        }
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ(tidx[static_cast<std::size_t>(j)], j);
            gt[j] += tg[j];
        }
        Ht += th;
    }
    const double h = 1e-4;
    for (int j = 0; j < 36; ++j) {
        auto plus = f, minus = f;
        plus.u[j] += h;
        minus.u[j] -= h;
        EXPECT_NEAR(gu[j], (total(plus) - total(minus)) / (2 * h), 1e-7 * (1 + gu.norm()));
    }
    for (int j = 0; j < 18; ++j) {
        auto plus = f, minus = f;
        plus.fp[j] += h;
        minus.fp[j] -= h;
        EXPECT_NEAR(gp[j], (total(plus) - total(minus)) / (2 * h), 1e-7 * (1 + gp.norm()));
    }
    for (int j = 0; j < 2; ++j) {
        auto plus = f, minus = f;
        plus.theta[j] += h;
        minus.theta[j] -= h;
        EXPECT_NEAR(gt[j], (total(plus) - total(minus)) / (2 * h), 1e-9);
    }
    EXPECT_TRUE(Hp.topRightCorner(9, 9).isZero(0));
    EXPECT_TRUE(Ht.isZero(0));
    const auto before = f.fields(0);
    const Eigen::Matrix3d original = f.fp.head<9>().reshaped(3, 3);
    f.fp.head<9>() = 1.1 * original.reshaped();
    f.theta << 4, 5;
    const Eigen::Vector3d xi(0.2, 0.3, 0.4);
    EXPECT_EQ(before.Fp.value(xi), original);
    EXPECT_EQ(before.theta.value(xi), Eigen::Vector2d(2, 3));
    EXPECT_NE(f.fields(0).Fp.value(xi), original);
    EXPECT_EQ(f.fields(1).theta.value(xi), Eigen::Vector2d(4, 5));
}

TEST(MaterialSources, SharedParameterHessianIncludesBothElements) {
    const Fixture f;
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    double volume = 0;
    for (int e = 0; e < 2; ++e) {
        const auto X_field = f.X_field(e);
        const auto fields = f.fields(e);
        Eigen::MatrixXd local_hessian;
        integrate_volume<GaussLegendre3D<2>>(
            bind_energy<QuadraticParameterMaterial<2>, MaterialVariable::Theta>(X_field, fields), nullptr,
            &local_hessian);
        H += local_hessian;
        for (int q = 0; q < GaussLegendre3D<2>::num_points; ++q)
            volume += GaussLegendre3D<2>::weights()[q] *
                      litepgo::ReferenceMapping(X_field.evaluate(GaussLegendre3D<2>::points().row(q).transpose()))
                          .det_dX_dxi();
        check_fd(
            X_field, ConstantDofMap(f.topology, 2).gather(0, f.theta),
            [&](const Eigen::Vector2d& theta) {
                return MaterialSources{fields.F, fields.Fp, IdentityField<Eigen::VectorXd>(theta)};
            },
            [&](const auto& local_fields) {
                return bind_energy<QuadraticParameterMaterial<2>, MaterialVariable::Theta>(X_field, local_fields);
            });
    }
    EXPECT_TRUE(H.isApprox(volume * Eigen::Matrix2d::Identity(), 1e-13));
}

TEST(MaterialSources, DirectInfeasiblePlasticStateReachesMaterialValidation) {
    const Fixture f;
    auto fields = f.fields(0);
    Eigen::Matrix3d inverted = Eigen::Matrix3d::Identity();
    inverted(0, 0) = -1;
    fields.Fp = IdentityField<Eigen::Matrix3d>(inverted);
    const auto X_field = f.X_field(0);
    Eigen::VectorXd g;
    Eigen::MatrixXd h;
    EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Fp>(X_field, fields)),
                 InfeasibleState);
    EXPECT_THROW(
        integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Fp>(X_field, fields), &g),
        InfeasibleState);
    EXPECT_THROW(integrate_volume<GaussLegendre3D<2>>(bind_energy<NeoHookean, MaterialVariable::Fp>(X_field, fields),
                                                      nullptr, &h),
                 InfeasibleState);
}
} // namespace
