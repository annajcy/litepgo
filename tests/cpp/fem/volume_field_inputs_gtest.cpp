#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/constant_field.hpp"
#include "litepgo/fem/deformation_gradient_field.hpp"
#include "litepgo/fem/element_field.hpp"
#include "litepgo/fem/gauss_legendre_3d.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/interpolated_value_field.hpp"
#include "litepgo/fem/local_kernel.hpp"
#include "litepgo/fem/neo_hookean_local_energy.hpp"
#include "litepgo/fem/vertex_element_dof_map.hpp"
#include "litepgo/fem/volume_field_inputs.hpp"
#include "litepgo/fem/volume_quadrature_cache.hpp"
#include "quadratic_parameter_local_energy.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {
using namespace litepgo;
struct Fixture {
    HexTopology topology{12, {{0, 1, 2, 3, 4, 5, 6, 7}, {1, 8, 9, 2, 5, 10, 11, 6}}};
    Eigen::Matrix<double, 12, 3> X;
    VertexDofData<3> u{12};
    ElementDofData<Mat3> fp{2};
    ConstantDofData<Eigen::Vector2d> theta{Eigen::Vector2d(2, 3)};

    Fixture() {
        for (int e = 0; e < 2; ++e) {
            const auto& v = topology.cell_vertices(e);
            for (int i = 0; i < 8; ++i) {
                const Eigen::Vector3d p = HexQ1Basis::reference_node(i) + Eigen::Vector3d(e, 0, 0);
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
    VolumeQuadratureCache<HexQ1Basis, GaussLegendre3D<2>> cache(int e) const {
        VolumeReferenceGeometry<HexQ1Basis>::GeometryCoefficients local;
        const auto& v = topology.cell_vertices(e);
        for (int i = 0; i < 8; ++i)
            local.row(i) = X.row(v[static_cast<std::size_t>(i)]);
        return VolumeQuadratureCache<HexQ1Basis, GaussLegendre3D<2>>(VolumeReferenceGeometry<HexQ1Basis>(local));
    }
    auto fields(int e) const {
        const VertexElementDofMap map(topology);
        return VolumeFieldInputs{DeformationGradientField<HexQ1Basis>(map.gather(e, u)), ElementField(fp, e),
                                 ConstantField(theta)};
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
        Eigen::Matrix<double, NumDofs, 1> dg;
        Eigen::Matrix<double, NumDofs, NumDofs> dH;
        for (int j = 0; j < NumDofs; ++j) {
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
    Eigen::Matrix<double, NumDofs, 1> d, v;
    for (int j = 0; j < NumDofs; ++j) {
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

TEST(VolumeFieldInputs, MixedRepresentationsPointInputsAndLocalFiniteDifferences) {
    const Fixture f;
    const VertexElementDofMap map(f.topology);
    for (int e = 0; e < 2; ++e) {
        SCOPED_TRACE(e);
        const auto cache = f.cache(e);
        const auto fields = f.fields(e);
        const auto ue = map.gather(e, f.u);
        for (int q = 0; q < VolumeQuadratureCache<HexQ1Basis, GaussLegendre3D<2>>::num_quadrature_points; ++q) {
            const auto& ref = cache.sample(q).point;
            const auto point = fields.evaluate(ref);
            EXPECT_TRUE(point.F.isApprox(Mat3::Identity() + FieldInterpolation<HexQ1Basis, 3>(ue).gradient(ref)));
            EXPECT_EQ(point.Fp, f.fp.value(e));
            EXPECT_EQ(point.theta, f.theta.value());
            EXPECT_TRUE(fields.Fp.jacobian(ref).isIdentity());
            EXPECT_TRUE(fields.theta.jacobian(ref).isIdentity());
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
                return VolumeFieldInputs{DeformationGradientField<HexQ1Basis>(c), ElementField(f.fp, e),
                                         ConstantField(f.theta)};
            },
            [](const auto& local_fields) { return NeoHookeanLocalEnergyF(local_fields); });
        const Mat3 fp = f.fp.value(e);
        const Eigen::Matrix<double, 9, 1> pvec = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(fp.data());
        check_fd(
            kernel, pvec,
            [&](const Eigen::Matrix<double, 9, 1>& x) {
                return VolumeFieldInputs{DeformationGradientField<HexQ1Basis>(ue),
                                         ElementField<Mat3>(Eigen::Map<const Mat3>(x.data())), ConstantField(f.theta)};
            },
            [](const auto& local_fields) { return NeoHookeanLocalEnergyFp(local_fields); });
        check_fd(
            kernel, f.theta.value(),
            [&](const Eigen::Matrix<double, 2, 1>& x) {
                return VolumeFieldInputs{DeformationGradientField<HexQ1Basis>(ue), ElementField<Mat3>(fp),
                                         ConstantField<Eigen::Vector2d>(x)};
            },
            [](const auto& local_fields) { return NeoHookeanLocalEnergyTheta(local_fields); });
    }
}

TEST(VolumeFieldInputs, SameKernelAcceptsDifferentActiveDofCounts) {
    Fixture fixture;
    fixture.u.values().setZero();
    const auto cache = fixture.cache(0);
    const auto nodal = fixture.fields(0);
    const VolumeFieldInputs direct{ElementField<Mat3>(Mat3::Identity()), nodal.Fp, nodal.theta};
    const LocalKernel kernel(cache);
    const auto nodal_gradient = kernel.gradient(NeoHookeanLocalEnergyF(nodal));
    const auto direct_gradient = kernel.gradient(NeoHookeanLocalEnergyF(direct));
    static_assert(decltype(nodal_gradient)::RowsAtCompileTime == 24);
    static_assert(decltype(direct_gradient)::RowsAtCompileTime == 9);
    EXPECT_DOUBLE_EQ(kernel.value(NeoHookeanLocalEnergyF(nodal)), kernel.value(NeoHookeanLocalEnergyF(direct)));
    Eigen::Matrix<double, 24, 1> expected_gradient = Eigen::Matrix<double, 24, 1>::Zero();
    Eigen::Matrix<double, 24, 24> expected_hessian = Eigen::Matrix<double, 24, 24>::Zero();
    Mat9 direct_hessian = Mat9::Zero();
    for (int q = 0; q < decltype(cache)::num_quadrature_points; ++q) {
        const auto& sample = cache.sample(q);
        const auto inputs = nodal.evaluate(sample.point);
        const auto J = nodal.F.jacobian(sample.point);
        const auto stress = NeoHookeanEnergyF{}.gradient(inputs);
        const auto tangent = NeoHookeanEnergyF{}.hessian(inputs);
        expected_gradient +=
            sample.volume_weight * J.transpose() * Eigen::Map<const Eigen::Matrix<double, 9, 1>>(stress.data());
        expected_hessian += sample.volume_weight * J.transpose() * tangent * J;
        direct_hessian += sample.volume_weight * tangent;
    }
    EXPECT_TRUE(nodal_gradient.isApprox(expected_gradient, 1e-13));
    EXPECT_TRUE(kernel.hessian(NeoHookeanLocalEnergyF(nodal)).isApprox(expected_hessian, 1e-13));
    EXPECT_TRUE(kernel.hessian(NeoHookeanLocalEnergyF(direct)).isApprox(direct_hessian, 1e-13));
    const Eigen::Matrix<double, 9, 1> identity =
        Eigen::Map<const Eigen::Matrix<double, 9, 1>>(Mat3::Identity().eval().data());
    check_fd(
        kernel, identity,
        [&](const Eigen::Matrix<double, 9, 1>& x) {
            return VolumeFieldInputs{ElementField<Mat3>(Eigen::Map<const Mat3>(x.data())), nodal.Fp, nodal.theta};
        },
        [](const auto& fields) { return NeoHookeanLocalEnergyF(fields); });
}

TEST(VolumeFieldInputs, ElementAndSharedDofNumbersAccumulateIntoDifferentGlobalBlocks) {
    Fixture f;
    const VertexElementDofMap map(f.topology);
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
        const auto uidx = map.dof_indices(e, f.u);
        const auto pidx = f.fp.dof_indices(e);
        const auto tidx = f.theta.dof_indices();
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

TEST(VolumeFieldInputs, SharedParameterHessianIncludesBothElements) {
    const Fixture f;
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    double volume = 0;
    for (int e = 0; e < 2; ++e) {
        const auto cache = f.cache(e);
        const litepgo::LocalKernel kernel(cache);
        const auto fields = f.fields(e);
        H += kernel.hessian(QuadraticParameterLocalEnergy(fields));
        for (int q = 0; q < VolumeQuadratureCache<HexQ1Basis, GaussLegendre3D<2>>::num_quadrature_points; ++q)
            volume += cache.sample(q).volume_weight;
        check_fd(
            kernel, f.theta.value(),
            [&](const Eigen::Vector2d& theta) {
                return VolumeFieldInputs{fields.F, fields.Fp, ConstantField<Eigen::Vector2d>(theta)};
            },
            [](const auto& local_fields) { return QuadraticParameterLocalEnergy(local_fields); });
    }
    EXPECT_TRUE(H.isApprox(volume * Eigen::Matrix2d::Identity(), 1e-13));
}

TEST(VolumeFieldInputs, DirectInfeasiblePlasticStateReachesMaterialValidation) {
    const Fixture f;
    auto fields = f.fields(0);
    Mat3 inverted = Mat3::Identity();
    inverted(0, 0) = -1;
    fields.Fp = ElementField<Mat3>(inverted);
    const litepgo::LocalKernel kernel(f.cache(0));
    EXPECT_THROW(kernel.value(NeoHookeanLocalEnergyFp(fields)), InfeasibleState);
    EXPECT_THROW(kernel.gradient(NeoHookeanLocalEnergyFp(fields)), InfeasibleState);
    EXPECT_THROW(kernel.hessian(NeoHookeanLocalEnergyFp(fields)), InfeasibleState);
}
} // namespace
