#include "../local/quadratic_parameter_local_energy.hpp"
#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/dof/element_dof_map.hpp"
#include "litepgo/dof/vertex_dof_map.hpp"
#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/basis/hex_q1_basis.hpp"
#include "litepgo/fem/field/deformation_gradient_field.hpp"
#include "litepgo/fem/field/identity_field.hpp"
#include "litepgo/fem/field/interpolated_value_field.hpp"
#include "litepgo/fem/field/material_fields.hpp"
#include "litepgo/fem/geometry/volume_quadrature_cache.hpp"
#include "litepgo/fem/local/local_kernel.hpp"
#include "litepgo/fem/local/neo_hookean_local_energy.hpp"
#include "litepgo/fem/material/neo_hookean.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/tensor_product_quadrature.hpp"

#include "litepgo/fem/geometry/volume_reference_geometry.hpp"
#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {
using namespace litepgo;
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
    litepgo::VolumeQuadratureCache cache(int e) const {
        Eigen::Matrix<double, 8, 3> local;
        const auto& v = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i)
            local.row(i) = X.row(v[static_cast<std::size_t>(i)]);
        return litepgo::VolumeQuadratureCache(VolumeReferenceGeometry<HexQ1Basis>(local),
                                              litepgo::GaussLegendre3D<2>{});
    }
    auto fields(int e) const {
        const VertexDofMap map(topology, 3);
        return MaterialFields{
            std::make_shared<DeformationGradientField>(map.gather(e, u)),
            std::make_shared<IdentityField<Eigen::Matrix3d>>(ElementDofMap(topology, 9).gather(e, fp).reshaped(3, 3)),
            std::make_shared<IdentityField<Eigen::VectorXd>>(ConstantDofMap(topology, 2).gather(e, theta))};
    }
};

template <class Kernel, int NumDofs, class MakeFields, class BindEnergy>
void check_fd(const Kernel& kernel, const Eigen::Matrix<double, NumDofs, 1>& x, MakeFields make_fields,
              BindEnergy bind_energy) {
    const auto fields = make_fields(x);
    const auto g = kernel.gradient(bind_energy(fields));
    const auto H = kernel.hessian(bind_energy(fields));
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    for (double h : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(h);
        Eigen::Matrix<double, NumDofs, 1> dg(x.size());
        Eigen::Matrix<double, NumDofs, NumDofs> dH(x.size(), x.size());
        for (int j = 0; j < x.size(); ++j) {
            auto plus = x, minus = x;
            plus[j] += h;
            minus[j] -= h;
            dg[j] = (kernel.value(bind_energy(make_fields(plus))) - kernel.value(bind_energy(make_fields(minus)))) /
                    (2 * h);
            const auto plus_fields = make_fields(plus), minus_fields = make_fields(minus);
            dH.col(j) =
                (kernel.gradient(bind_energy(plus_fields)) - kernel.gradient(bind_energy(minus_fields))) / (2 * h);
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
    const double fd = (kernel.value(bind_energy(make_fields(x + h * d + h * v))) -
                       kernel.value(bind_energy(make_fields(x + h * d - h * v))) -
                       kernel.value(bind_energy(make_fields(x - h * d + h * v))) +
                       kernel.value(bind_energy(make_fields(x - h * d - h * v)))) /
                      (4 * h * h);
    EXPECT_NEAR(fd, d.dot(H * v), 2e-6 * (1 + H.norm()));
}

TEST(MaterialFields, MixedRepresentationsPointInputsAndLocalFiniteDifferences) {
    const Fixture f;
    const VertexDofMap map(f.topology, 3);
    for (int e = 0; e < 2; ++e) {
        SCOPED_TRACE(e);
        const auto cache = f.cache(e);
        const auto fields = f.fields(e);
        const auto ue = map.gather(e, f.u);
        for (int q = 0; q < cache.num_quadrature_points(); ++q) {
            const auto& ref = cache.sample(q).point;
            const auto point = fields.evaluate(ref);
            EXPECT_TRUE(point.F.isApprox(Eigen::Matrix3d::Identity() + FieldInterpolation(ue).gradient(ref)));
            EXPECT_EQ(point.Fp, f.fp.segment<9>(9 * e).reshaped(3, 3));
            EXPECT_EQ(point.theta, ConstantDofMap(f.topology, 2).gather(0, f.theta));
            EXPECT_TRUE(
                std::dynamic_pointer_cast<const AffineField<Eigen::Matrix3d>>(fields.Fp)->jacobian(ref).isIdentity());
            EXPECT_TRUE(std::dynamic_pointer_cast<const AffineField<Eigen::VectorXd>>(fields.theta)
                            ->jacobian(ref)
                            .isIdentity());
        }
        const litepgo::LocalKernel kernel(cache);
        EXPECT_DOUBLE_EQ(kernel.value(NeoHookeanLocalEnergyF(fields)), kernel.value(NeoHookeanLocalEnergyFp(fields)));
        EXPECT_DOUBLE_EQ(kernel.value(NeoHookeanLocalEnergyF(fields)),
                         kernel.value(NeoHookeanLocalEnergyTheta(fields)));
        const Eigen::Matrix<double, 24, 1> uvec = ue.reshaped<Eigen::RowMajor>();
        check_fd(
            kernel, uvec,
            [&](const Eigen::Matrix<double, 24, 1>& x) {
                Eigen::Matrix<double, 8, 3> c;
                for (int j = 0; j < 24; ++j)
                    c(j / 3, j % 3) = x[j];
                return MaterialFields{
                    std::make_shared<DeformationGradientField>(c),
                    std::make_shared<IdentityField<Eigen::Matrix3d>>(
                        ElementDofMap(f.topology, 9).gather(e, f.fp).reshaped(3, 3)),
                    std::make_shared<IdentityField<Eigen::VectorXd>>(ConstantDofMap(f.topology, 2).gather(e, f.theta))};
            },
            [](const auto& local_fields) { return NeoHookeanLocalEnergyF(local_fields); });
        const Eigen::Matrix3d fp = f.fp.segment<9>(9 * e).reshaped(3, 3);
        const Eigen::Matrix<double, 9, 1> pvec = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fp.data());
        check_fd(
            kernel, pvec,
            [&](const Eigen::Matrix<double, 9, 1>& x) {
                return MaterialFields{
                    std::make_shared<DeformationGradientField>(ue),
                    std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Map<const Eigen::Matrix3d>(x.data())),
                    std::make_shared<IdentityField<Eigen::VectorXd>>(ConstantDofMap(f.topology, 2).gather(e, f.theta))};
            },
            [](const auto& local_fields) { return NeoHookeanLocalEnergyFp(local_fields); });
        check_fd(
            kernel, ConstantDofMap(f.topology, 2).gather(0, f.theta),
            [&](const Eigen::Matrix<double, 2, 1>& x) {
                return MaterialFields{std::make_shared<DeformationGradientField>(ue),
                                      std::make_shared<IdentityField<Eigen::Matrix3d>>(fp),
                                      std::make_shared<IdentityField<Eigen::VectorXd>>(x)};
            },
            [](const auto& local_fields) { return NeoHookeanLocalEnergyTheta(local_fields); });
    }
}

TEST(MaterialFields, SameKernelAcceptsDifferentActiveDofCounts) {
    Fixture fixture;
    fixture.u.setZero();
    const auto cache = fixture.cache(0);
    const auto nodal = fixture.fields(0);
    const MaterialFields direct{std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Matrix3d::Identity()), nodal.Fp,
                                nodal.theta};
    const LocalKernel kernel(cache);
    const auto nodal_gradient = kernel.gradient(NeoHookeanLocalEnergyF(nodal));
    const auto direct_gradient = kernel.gradient(NeoHookeanLocalEnergyF(direct));
    EXPECT_EQ(nodal_gradient.size(), 24);
    EXPECT_EQ(direct_gradient.size(), 9);
    EXPECT_DOUBLE_EQ(kernel.value(NeoHookeanLocalEnergyF(nodal)), kernel.value(NeoHookeanLocalEnergyF(direct)));
    Eigen::Matrix<double, 24, 1> expected_gradient = Eigen::Matrix<double, 24, 1>::Zero();
    Eigen::Matrix<double, 24, 24> expected_hessian = Eigen::Matrix<double, 24, 24>::Zero();
    Eigen::Matrix<double, 9, 9> direct_hessian = Eigen::Matrix<double, 9, 9>::Zero();
    for (int q = 0; q < cache.num_quadrature_points(); ++q) {
        const auto& sample = cache.sample(q);
        const auto inputs = nodal.evaluate(sample.point);
        const auto J = std::dynamic_pointer_cast<const AffineField<Eigen::Matrix3d>>(nodal.F)->jacobian(sample.point);
        const auto stress = NeoHookean{}.gradient_F(inputs);
        const auto tangent = NeoHookean{}.hessian_F(inputs);
        expected_gradient +=
            sample.volume_weight * J.transpose() * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(stress.data());
        expected_hessian += sample.volume_weight * J.transpose() * tangent * J;
        direct_hessian += sample.volume_weight * tangent;
    }
    EXPECT_TRUE(nodal_gradient.isApprox(expected_gradient, 1e-13));
    EXPECT_TRUE(kernel.hessian(NeoHookeanLocalEnergyF(nodal)).isApprox(expected_hessian, 1e-13));
    EXPECT_TRUE(kernel.hessian(NeoHookeanLocalEnergyF(direct)).isApprox(direct_hessian, 1e-13));
    const Eigen::Matrix<double, 9, 1> identity =
        Eigen::Map<const Eigen::Matrix<double, 9, 1>>(Eigen::Matrix3d::Identity().eval().data());
    check_fd(
        kernel, identity,
        [&](const Eigen::Matrix<double, 9, 1>& x) {
            return MaterialFields{
                std::make_shared<IdentityField<Eigen::Matrix3d>>(Eigen::Map<const Eigen::Matrix3d>(x.data())), nodal.Fp,
                nodal.theta};
        },
        [](const auto& fields) { return NeoHookeanLocalEnergyF(fields); });
}

TEST(MaterialFields, ElementAndSharedDofNumbersAccumulateIntoDifferentGlobalBlocks) {
    Fixture f;
    const VertexDofMap map(f.topology, 3);
    std::array caches{f.cache(0), f.cache(1)};
    const auto total = [&](const Fixture& data) {
        double value = 0;
        for (int e = 0; e < 2; ++e)
            value += LocalKernel(caches[static_cast<std::size_t>(e)]).value(NeoHookeanLocalEnergyF(data.fields(e)));
        return value;
    };
    Vec gu = Vec::Zero(36), gp = Vec::Zero(18), gt = Vec::Zero(2);
    Eigen::MatrixXd Hp = Eigen::MatrixXd::Zero(18, 18);
    Eigen::Matrix2d Ht = Eigen::Matrix2d::Zero();
    for (int e = 0; e < 2; ++e) {
        const auto& cache = caches[static_cast<std::size_t>(e)];
        const auto fields = f.fields(e);
        const auto uidx = map.dof_indices(e);
        const auto pidx = ElementDofMap(f.topology, 9).dof_indices(e);
        const auto tidx = ConstantDofMap(f.topology, 2).dof_indices(e);
        const auto ug = LocalKernel(cache).gradient(NeoHookeanLocalEnergyF(fields));
        const litepgo::LocalKernel kernel(cache);
        const auto pg = kernel.gradient(NeoHookeanLocalEnergyFp(fields));
        const auto ph = kernel.hessian(NeoHookeanLocalEnergyFp(fields));
        const auto tg = kernel.gradient(NeoHookeanLocalEnergyTheta(fields));
        const auto th = kernel.hessian(NeoHookeanLocalEnergyTheta(fields));
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
    EXPECT_EQ(before.Fp->value(caches[0].sample(0).point), original);
    EXPECT_EQ(before.theta->value(caches[0].sample(0).point), Eigen::Vector2d(2, 3));
    EXPECT_NE(f.fields(0).Fp->value(caches[0].sample(0).point), original);
    EXPECT_EQ(f.fields(1).theta->value(caches[1].sample(0).point), Eigen::Vector2d(4, 5));
}

TEST(MaterialFields, SharedParameterHessianIncludesBothElements) {
    const Fixture f;
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    double volume = 0;
    for (int e = 0; e < 2; ++e) {
        const auto cache = f.cache(e);
        const litepgo::LocalKernel kernel(cache);
        const auto fields = f.fields(e);
        H += kernel.hessian(QuadraticParameterLocalEnergy(fields));
        for (int q = 0; q < cache.num_quadrature_points(); ++q)
            volume += cache.sample(q).volume_weight;
        check_fd(
            kernel, ConstantDofMap(f.topology, 2).gather(0, f.theta),
            [&](const Eigen::Vector2d& theta) {
                return MaterialFields{fields.F, fields.Fp, std::make_shared<IdentityField<Eigen::VectorXd>>(theta)};
            },
            [](const auto& local_fields) { return QuadraticParameterLocalEnergy(local_fields); });
    }
    EXPECT_TRUE(H.isApprox(volume * Eigen::Matrix2d::Identity(), 1e-13));
}

TEST(MaterialFields, DirectInfeasiblePlasticStateReachesMaterialValidation) {
    const Fixture f;
    auto fields = f.fields(0);
    Eigen::Matrix3d inverted = Eigen::Matrix3d::Identity();
    inverted(0, 0) = -1;
    fields.Fp = std::make_shared<IdentityField<Eigen::Matrix3d>>(inverted);
    const litepgo::LocalKernel kernel(f.cache(0));
    EXPECT_THROW(kernel.value(NeoHookeanLocalEnergyFp(fields)), InfeasibleState);
    EXPECT_THROW(kernel.gradient(NeoHookeanLocalEnergyFp(fields)), InfeasibleState);
    EXPECT_THROW(kernel.hessian(NeoHookeanLocalEnergyFp(fields)), InfeasibleState);
}
} // namespace
