#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/constant_field.hpp"
#include "litepgo/fem/direct_local_field.hpp"
#include "litepgo/fem/displacement_gradient_field.hpp"
#include "litepgo/fem/element_field.hpp"
#include "litepgo/fem/gauss_legendre_3d.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/interpolated_value_field.hpp"
#include "litepgo/fem/neo_hookean.hpp"
#include "litepgo/fem/vertex_element_dof_map.hpp"
#include "litepgo/fem/volume_field_mapping.hpp"
#include "litepgo/fem/volume_local_kernel.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {
using namespace litepgo;
using Basis = HexQ1Basis;
using Geometry = VolumeReferenceGeometry<Basis>;
using Ref = Geometry::PointData;
using Cache = VolumeQuadratureCache<Basis, GaussLegendre3D<2>>;
using U = DisplacementGradientField<Basis>;
using P = DirectLocalField<Mat3>;
using T = DirectLocalField<Eigen::Vector2d>;
using Fields = VolumeFieldInputs<U, P, T>;
using FM = VolumeFieldMapping<Ref, Fields, &Fields::F>;
using PM = VolumeFieldMapping<Ref, Fields, &Fields::Fp>;
using TM = VolumeFieldMapping<Ref, Fields, &Fields::theta>;
using FK = LocalKernel<FM, Cache, NeoHookeanEnergyF>;
using PK = LocalKernel<PM, Cache, NeoHookeanEnergyFp>;
using TK = LocalKernel<TM, Cache, NeoHookeanEnergyTheta>;
static_assert(FK::num_dofs == 24 && PK::num_dofs == 9 && TK::num_dofs == 2);
static_assert(std::same_as<Fields::Point, VolumetricPoint<2>>);

struct Fixture {
    HexTopology topology{12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}}};
    Eigen::Matrix<double, 12, 3> X;
    VertexDofData<3> u{12};
    ElementField<Mat3> fp{2};
    ConstantField<Eigen::Vector2d> theta{Eigen::Vector2d(2, 3)};

    Fixture() {
        for (int e = 0; e < 2; ++e) {
            const auto& v = topology.cell_vertices(e);
            for (int i = 0; i < 8; ++i) {
                const Eigen::Vector3d p = Basis::reference_node(i) + Eigen::Vector3d(e, 0, 0);
                X.row(v[static_cast<std::size_t>(i)]) = Eigen::RowVector3d(
                    1.2 * p.x() + 0.1 * p.y() * p.z(), 0.9 * p.y() + 0.05 * p.x() * p.z(), 1.1 * p.z());
            }
            Mat3 p;
            p << 1.1 + 0.1 * e, 0.08, -0.03, 0.02, 0.9 + 0.05 * e, 0.04, -0.01, 0.03, 1.05;
            fp.set_value(e, p);
        }
        for (int j = 0; j < u.num_dofs(); ++j)
            u.values()[j] = 0.04 * std::sin(0.7 * (j + 1));
    }
    Cache cache(int e) const {
        Geometry::GeometryCoefficients local;
        const auto& v = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i)
            local.row(i) = X.row(v[static_cast<std::size_t>(i)]);
        return Cache(Geometry(local));
    }
    Fields fields(int e) const {
        const VertexElementDofMap map(topology);
        return {U(map.gather(e, u)), P(fp.value(e)), T(theta.value())};
    }
};

template <class K, class MakeFields>
void check_fd(const K& kernel, const typename K::Gradient& x, MakeFields make_fields) {
    const auto fields = make_fields(x);
    const auto g = kernel.gradient(fields);
    const auto H = kernel.hessian(fields);
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    for (double h : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(h);
        typename K::Gradient dg;
        typename K::Hessian dH;
        for (int j = 0; j < K::num_dofs; ++j) {
            auto plus = x, minus = x;
            plus[j] += h;
            minus[j] -= h;
            dg[j] = (kernel.value(make_fields(plus)) - kernel.value(make_fields(minus))) / (2 * h);
            dH.col(j) = (kernel.gradient(make_fields(plus)) - kernel.gradient(make_fields(minus))) / (2 * h);
        }
        EXPECT_LT((g - dg).norm() / (1 + g.norm()), 20 * h * h + 1e-9);
        EXPECT_LT((H - dH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
    }
    typename K::Gradient d, v;
    for (int j = 0; j < K::num_dofs; ++j) {
        d[j] = std::sin(j + 0.5);
        v[j] = std::cos(j + 0.2);
    }
    d.normalize();
    v.normalize();
    const double h = 3e-4;
    const double fd = (kernel.value(make_fields(x + h * d + h * v)) - kernel.value(make_fields(x + h * d - h * v)) -
                       kernel.value(make_fields(x - h * d + h * v)) + kernel.value(make_fields(x - h * d - h * v))) /
                      (4 * h * h);
    EXPECT_NEAR(fd, d.dot(H * v), 2e-6 * (1 + H.norm()));
}

TEST(VolumeFieldMapping, MixedRepresentationsPointInputsAndLocalFiniteDifferences) {
    const Fixture f;
    const VertexElementDofMap map(f.topology);
    for (int e = 0; e < 2; ++e) {
        SCOPED_TRACE(e);
        const auto cache = f.cache(e);
        const auto fields = f.fields(e);
        const auto ue = map.gather(e, f.u);
        for (int q = 0; q < Cache::num_quadrature_points; ++q) {
            const auto& ref = cache.sample(q).point;
            const auto point = FM{}.point_input(fields, ref);
            const auto ppoint = PM{}.point_input(fields, ref);
            const auto tpoint = TM{}.point_input(fields, ref);
            EXPECT_TRUE(point.F.isApprox(Mat3::Identity() + InterpolatedField<Basis, 3>(ue).gradient(ref)));
            EXPECT_EQ(point.Fp, f.fp.value(e));
            EXPECT_EQ(point.theta, f.theta.value());
            EXPECT_EQ(ppoint.F, point.F);
            EXPECT_EQ(tpoint.Fp, point.Fp);
            EXPECT_TRUE(PM{}.active_jacobian(ref).isIdentity());
            EXPECT_TRUE(TM{}.active_jacobian(ref).isIdentity());
        }
        const FK fk(cache, FM{}, NeoHookeanEnergyF{});
        const PK pk(cache, PM{}, NeoHookeanEnergyFp{});
        const TK tk(cache, TM{}, NeoHookeanEnergyTheta{});
        EXPECT_DOUBLE_EQ(fk.value(fields), pk.value(fields));
        EXPECT_DOUBLE_EQ(fk.value(fields), tk.value(fields));
        const FK::Gradient uvec = ue.reshaped<Eigen::RowMajor>();
        check_fd(fk, uvec, [&](const FK::Gradient& x) {
            U::Coefficients c;
            for (int j = 0; j < 24; ++j)
                c(j / 3, j % 3) = x[j];
            return Fields{U(c), P(f.fp.value(e)), T(f.theta.value())};
        });
        const Mat3 fp = f.fp.value(e);
        const PK::Gradient pvec = Eigen::Map<const PK::Gradient>(fp.data());
        check_fd(pk, pvec, [&](const PK::Gradient& x) {
            return Fields{U(ue), P(Eigen::Map<const Mat3>(x.data())), T(f.theta.value())};
        });
        check_fd(tk, f.theta.value(), [&](const TK::Gradient& x) { return Fields{U(ue), P(fp), T(x)}; });
    }
}

TEST(VolumeFieldMapping, ElementAndSharedDofNumbersAccumulateIntoDifferentGlobalBlocks) {
    Fixture f;
    const VertexElementDofMap map(f.topology);
    std::array<Cache, 2> caches{f.cache(0), f.cache(1)};
    const auto total = [&](const Fixture& data) {
        double value = 0;
        for (int e = 0; e < 2; ++e)
            value += FK(caches[static_cast<std::size_t>(e)], FM{}, NeoHookeanEnergyF{}).value(data.fields(e));
        return value;
    };
    Vec gu = Vec::Zero(36), gp = Vec::Zero(18), gt = Vec::Zero(2);
    Eigen::MatrixXd Hp = Eigen::MatrixXd::Zero(18, 18);
    Eigen::Matrix2d Ht = Eigen::Matrix2d::Zero();
    for (int e = 0; e < 2; ++e) {
        const auto& cache = caches[static_cast<std::size_t>(e)];
        const auto fields = f.fields(e);
        const auto uidx = map.dof_indices(e, f.u);
        const auto pidx = f.fp.dof_indices(e);
        const auto tidx = f.theta.dof_indices();
        const auto ug = FK(cache, FM{}, NeoHookeanEnergyF{}).gradient(fields);
        const PK pk(cache, PM{}, NeoHookeanEnergyFp{});
        const auto pg = pk.gradient(fields);
        const auto ph = pk.hessian(fields);
        const TK tk(cache, TM{}, NeoHookeanEnergyTheta{});
        const auto tg = tk.gradient(fields);
        const auto th = tk.hessian(fields);
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
        plus.u.values()[j] += h;
        minus.u.values()[j] -= h;
        EXPECT_NEAR(gu[j], (total(plus) - total(minus)) / (2 * h), 1e-7 * (1 + gu.norm()));
    }
    for (int j = 0; j < 18; ++j) {
        auto plus = f, minus = f;
        plus.fp.values()[j] += h;
        minus.fp.values()[j] -= h;
        EXPECT_NEAR(gp[j], (total(plus) - total(minus)) / (2 * h), 1e-7 * (1 + gp.norm()));
    }
    for (int j = 0; j < 2; ++j) {
        auto plus = f, minus = f;
        plus.theta.values()[j] += h;
        minus.theta.values()[j] -= h;
        EXPECT_NEAR(gt[j], (total(plus) - total(minus)) / (2 * h), 1e-9);
    }
    EXPECT_TRUE(Hp.topRightCorner(9, 9).isZero(0));
    EXPECT_TRUE(Ht.isZero(0));
    const auto before = f.fields(0);
    const Mat3 original = f.fp.value(0);
    f.fp.set_value(0, 1.1 * original);
    f.theta.set_value({4, 5});
    EXPECT_EQ(before.Fp.value(caches[0].sample(0).point), original);
    EXPECT_EQ(before.theta.value(caches[0].sample(0).point), Eigen::Vector2d(2, 3));
    EXPECT_NE(f.fields(0).Fp.value(caches[0].sample(0).point), original);
    EXPECT_EQ(f.fields(1).theta.value(caches[1].sample(0).point), Eigen::Vector2d(4, 5));
}

struct QuadraticTheta {
    using Point = VolumetricPoint<2>;
    using Gradient = Eigen::Vector2d;
    using Hessian = Eigen::Matrix2d;
    double value(const Point& p) const {
        return 0.5 * p.theta.squaredNorm();
    }
    Gradient gradient(const Point& p) const {
        return p.theta;
    }
    Hessian hessian(const Point&) const {
        return Hessian::Identity();
    }
};
TEST(VolumeFieldMapping, SharedParameterHessianIncludesBothElements) {
    const Fixture f;
    using K = LocalKernel<TM, Cache, QuadraticTheta>;
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    double volume = 0;
    for (int e = 0; e < 2; ++e) {
        const auto cache = f.cache(e);
        const K kernel(cache, TM{}, QuadraticTheta{});
        H += kernel.hessian(f.fields(e));
        for (int q = 0; q < Cache::num_quadrature_points; ++q)
            volume += cache.sample(q).volume_weight;
        const auto fields = f.fields(e);
        check_fd(kernel, f.theta.value(),
                 [&](const Eigen::Vector2d& theta) { return Fields{fields.F, fields.Fp, T(theta)}; });
    }
    EXPECT_TRUE(H.isApprox(volume * Eigen::Matrix2d::Identity(), 1e-13));
}

TEST(VolumeFieldMapping, NodalParameterAndPlasticPathsMatchExistingKernels) {
    const Fixture f;
    using IV = InterpolatedValueField<Basis, 2>;
    using NF = VolumeFieldInputs<U, U, IV>;
    using NM = VolumeFieldMapping<Ref, NF, &NF::theta>;
    using K = LocalKernel<NM, Cache, NeoHookeanEnergyTheta>;
    const VertexElementDofMap map(f.topology);
    VolumeLocalFields<Basis, 2> old;
    old.u = map.gather(0, f.u);
    old.up = 0.3 * old.u;
    for (int i = 0; i < 8; ++i) {
        old.theta(i, 0) = 2 + 0.1 * i;
        old.theta(i, 1) = 3 + 0.2 * i;
    }
    const NF fields{U(old.u), U(old.up), IV(old.theta)};
    const auto cache = f.cache(0);
    const K kernel(cache, NM{}, NeoHookeanEnergyTheta{});
    const VolumeThetaKernel<Basis, GaussLegendre3D<2>, 2, NeoHookeanEnergyTheta> original(
        cache, VolumeThetaMapping<Basis, 2>{}, NeoHookeanEnergyTheta{});
    EXPECT_DOUBLE_EQ(kernel.value(fields), original.value(old));
    EXPECT_TRUE(kernel.gradient(fields).isApprox(original.gradient(old), 1e-13));
    EXPECT_EQ(kernel.hessian(fields), original.hessian(old));
}

TEST(VolumeFieldMapping, DirectInfeasiblePlasticStateReachesMaterialValidation) {
    const Fixture f;
    auto fields = f.fields(0);
    Mat3 inverted = Mat3::Identity();
    inverted(0, 0) = -1;
    fields.Fp = P(inverted);
    const PK kernel(f.cache(0), PM{}, NeoHookeanEnergyFp{});
    EXPECT_THROW(kernel.value(fields), InfeasibleState);
    EXPECT_THROW(kernel.gradient(fields), InfeasibleState);
    EXPECT_THROW(kernel.hessian(fields), InfeasibleState);
}
} // namespace
