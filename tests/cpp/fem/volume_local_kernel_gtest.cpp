#include "litepgo/energy/energy.hpp"
#include "litepgo/fem/gauss_legendre_3d.hpp"
#include "litepgo/fem/hex_q1_basis.hpp"
#include "litepgo/fem/neo_hookean.hpp"
#include "litepgo/fem/volume_local_kernel.hpp"

#include <Eigen/LU>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace litepgo;
using Basis = HexQ1Basis;
using Quad = GaussLegendre3D<2>;
using Geometry = VolumeReferenceGeometry<Basis>;
using Cache = VolumeQuadratureCache<Basis, Quad>;
using Fields = VolumeLocalFields<Basis, 2>;
using FK = VolumeFKernel<Basis, Quad, 2, NeoHookeanEnergyF>;
using PK = VolumeFpKernel<Basis, Quad, 2, NeoHookeanEnergyFp>;
using TK = VolumeThetaKernel<Basis, Quad, 2, NeoHookeanEnergyTheta>;

static_assert(std::same_as<VolumeFMapping<Basis, 2>, VolumeFpMapping<Basis, 2>>);
static_assert(FK::num_dofs == 24 && PK::num_dofs == 24 && TK::num_dofs == 16);
static_assert(std::same_as<Fields::VectorCoefficients, Eigen::Matrix<double, 8, 3>>);
static_assert(std::same_as<Fields::ParameterCoefficients, Eigen::Matrix<double, 8, 2>>);

template <class K>
concept HasGradient = requires(const K& k, const typename K::Fields& f) { k.gradient(f); };
template <class K>
concept HasHessian = requires(const K& k, const typename K::Fields& f) { k.hessian(f); };

struct ValueOnly {
    using Point = VolumetricPoint<2>;
    double value(const Point& p) const {
        return p.F.squaredNorm();
    }
};
struct FirstOnly : ValueOnly {
    using Gradient = Mat3;
    Gradient gradient(const Point& p) const {
        return 2 * p.F;
    }
};
struct SecondOnly : ValueOnly {
    using Hessian = Mat9;
    Hessian hessian(const Point&) const {
        return 2 * Mat9::Identity();
    }
};
using VK = VolumeFKernel<Basis, Quad, 2, ValueOnly>;
using GK = VolumeFKernel<Basis, Quad, 2, FirstOnly>;
using HK = VolumeFKernel<Basis, Quad, 2, SecondOnly>;
static_assert(!HasGradient<VK> && !HasHessian<VK>);
static_assert(HasGradient<GK> && !HasHessian<GK>);
static_assert(!HasGradient<HK> && HasHessian<HK>);
static_assert(!HasGradient<VolumeThetaKernel<Basis, Quad, 2, FirstOnly>>);
static_assert(!HasHessian<VolumeThetaKernel<Basis, Quad, 2, SecondOnly>>);
static_assert(!HasGradient<VolumeFpKernel<Basis, Quad, 2, ValueOnly>>);
static_assert(!HasHessian<VolumeFpKernel<Basis, Quad, 2, ValueOnly>>);

Geometry::GeometryCoefficients coordinates(bool warped = true) {
    Geometry::GeometryCoefficients X;
    Mat3 J;
    J << 1.7, 0.2, 0.1, 0.1, 1.2, 0.2, 0.0, 0.1, 0.9;
    for (int i = 0; i < 8; ++i) {
        const auto xi = Basis::reference_node(i);
        Eigen::Vector3d x = J * xi;
        if (warped)
            x += Eigen::Vector3d(0.15 * xi.y() * xi.z(), 0.1 * xi.x() * xi.z(), 0.05 * xi.x() * xi.y());
        X.row(i) = x.transpose();
    }
    return X;
}

Fields fields(int seed = 0) {
    Fields f;
    for (int i = 0; i < 8; ++i) {
        for (int c = 0; c < 3; ++c) {
            f.u(i, c) = 0.06 * std::sin(0.7 * (i * 3 + c + 1) * (seed + 1));
            f.up(i, c) = 0.035 * std::cos(0.4 * (i * 3 + c + 1) * (seed + 2));
        }
        f.theta(i, 0) = 2 + 0.15 * i;
        f.theta(i, 1) = 3 + 0.1 * std::sin(i + seed);
    }
    return f;
}

template <class K, class Member> void check_fd(const K& kernel, const Fields& f, Member member) {
    const auto G = kernel.gradient(f);
    const auto H = kernel.hessian(f);
    EXPECT_LT((H - H.transpose()).norm(), 1e-12 * (1 + H.norm()));
    const int components = static_cast<int>((f.*member).cols());
    for (double h : {1e-3, 3e-4, 1e-4}) {
        SCOPED_TRACE(h);
        typename K::Gradient fdG;
        typename K::Hessian fdH;
        for (int j = 0; j < K::num_dofs; ++j) {
            auto plus = f, minus = f;
            (plus.*member)(j / components, j % components) += h;
            (minus.*member)(j / components, j % components) -= h;
            fdG[j] = (kernel.value(plus) - kernel.value(minus)) / (2 * h);
            fdH.col(j) = (kernel.gradient(plus) - kernel.gradient(minus)) / (2 * h);
        }
        EXPECT_LT((G - fdG).norm() / (1 + G.norm()), 20 * h * h + 1e-9);
        EXPECT_LT((H - fdH).norm() / (1 + H.norm()), 20 * h * h + 1e-9);
    }
    // Independent mixed directional second difference of the scalar energy.
    typename K::Gradient d, e;
    for (int j = 0; j < K::num_dofs; ++j) {
        d[j] = std::sin(j + 0.3);
        e[j] = std::cos(0.7 * j + 0.2);
    }
    d.normalize();
    e.normalize();
    constexpr double h = 3e-4;
    auto pp = f, pm = f, mp = f, mm = f;
    for (int j = 0; j < K::num_dofs; ++j) {
        const int row = j / components, col = j % components;
        (pp.*member)(row, col) += h * (d[j] + e[j]);
        (pm.*member)(row, col) += h * (d[j] - e[j]);
        (mp.*member)(row, col) += h * (-d[j] + e[j]);
        (mm.*member)(row, col) -= h * (d[j] + e[j]);
    }
    const double fd = (kernel.value(pp) - kernel.value(pm) - kernel.value(mp) + kernel.value(mm)) / (4 * h * h);
    EXPECT_NEAR(fd, d.dot(H * e), 2e-6 * (1 + H.norm()));
}

TEST(VolumeLocalKernel, HexQ1LocalDofFiniteDifferencesForAllThreeFields) {
    const Cache cache{Geometry(coordinates())};
    const FK f_kernel(cache, VolumeFMapping<Basis, 2>{}, NeoHookeanEnergyF{});
    const PK p_kernel(cache, VolumeFpMapping<Basis, 2>{}, NeoHookeanEnergyFp{});
    const TK t_kernel(cache, VolumeThetaMapping<Basis, 2>{}, NeoHookeanEnergyTheta{});
    for (int seed = 0; seed < 3; ++seed) {
        SCOPED_TRACE(seed);
        const auto f = fields(seed);
        EXPECT_DOUBLE_EQ(f_kernel.value(f), p_kernel.value(f));
        EXPECT_DOUBLE_EQ(f_kernel.value(f), t_kernel.value(f));
        check_fd(f_kernel, f, &Fields::u);
        check_fd(p_kernel, f, &Fields::up);
        check_fd(t_kernel, f, &Fields::theta);
    }
}

TEST(VolumeLocalKernel, AffineFieldsHaveExpectedVolumeWeightedEnergyAndTranslationNullModes) {
    const auto X = coordinates(false);
    const Cache cache{Geometry(X)};
    const FK fk(cache, VolumeFMapping<Basis, 2>{}, NeoHookeanEnergyF{});
    const PK pk(cache, VolumeFpMapping<Basis, 2>{}, NeoHookeanEnergyFp{});
    Mat3 A, P;
    A << 0.1, 0.02, -0.03, 0.04, -0.05, 0.02, 0.01, 0.03, 0.08;
    P << 0.2, 0.03, 0, 0.01, 0.1, 0.02, 0, 0.01, -0.05;
    Fields f;
    f.u = X * A.transpose();
    f.up = X * P.transpose();
    f.theta.col(0).setConstant(2);
    f.theta.col(1).setConstant(3);
    Mat3 J;
    J.col(0) = X.row(1) - X.row(0);
    J.col(1) = X.row(3) - X.row(0);
    J.col(2) = X.row(4) - X.row(0);
    const double expected = J.determinant() * NeoHookean{}.value({Mat3::Identity() + A, Mat3::Identity() + P, {2, 3}});
    EXPECT_NEAR(fk.value(f), expected, 1e-12);
    for (int c = 0; c < 3; ++c) {
        FK::Gradient translation = FK::Gradient::Zero();
        for (int i = 0; i < 8; ++i)
            translation[3 * i + c] = 1;
        EXPECT_LT((fk.hessian(f) * translation).norm(), 1e-12);
        EXPECT_LT((pk.hessian(f) * translation).norm(), 1e-12);
        auto shifted = f;
        shifted.u.col(c).array() += 0.7;
        shifted.up.col(c).array() -= 0.4;
        EXPECT_NEAR(fk.value(shifted), expected, 1e-12);
    }
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
TEST(VolumeLocalKernel, NonzeroParameterHessianAndIndependentDerivativeCapabilities) {
    const Cache cache{Geometry(coordinates())};
    const auto f = fields();
    const VolumeThetaKernel<Basis, Quad, 2, QuadraticTheta> tk(cache, VolumeThetaMapping<Basis, 2>{}, QuadraticTheta{});
    EXPECT_GT(tk.hessian(f).norm(), 0.1);
    check_fd(tk, f, &Fields::theta);
    const VK vk(cache, VolumeFMapping<Basis, 2>{}, ValueOnly{});
    const GK gk(cache, VolumeFMapping<Basis, 2>{}, FirstOnly{});
    const HK hk(cache, VolumeFMapping<Basis, 2>{}, SecondOnly{});
    EXPECT_DOUBLE_EQ(vk.value(f), gk.value(f));
    EXPECT_DOUBLE_EQ(vk.value(f), hk.value(f));
    constexpr double h = 1e-4;
    for (int j = 0; j < 24; ++j) {
        auto plus = f, minus = f;
        plus.u(j / 3, j % 3) += h;
        minus.u(j / 3, j % 3) -= h;
        EXPECT_NEAR(gk.gradient(f)[j], (vk.value(plus) - vk.value(minus)) / (2 * h), 1e-9);
        EXPECT_LT((hk.hessian(f).col(j) - (gk.gradient(plus) - gk.gradient(minus)) / (2 * h)).norm(), 1e-9);
    }
}

struct EmptyTheta {
    using Point = VolumetricPoint<0>;
    using Gradient = MaterialParameterTypes<0>::Parameters;
    using Hessian = MaterialParameterTypes<0>::ParameterHessian;
    double value(const Point&) const {
        return 1.0;
    }
    Gradient gradient(const Point&) const {
        return {};
    }
    Hessian hessian(const Point&) const {
        return {};
    }
};
TEST(VolumeLocalKernel, ZeroParametersDoNotInstantiateAnEmptyInterpolatedField) {
    const Cache cache{Geometry(coordinates(false))};
    VolumeLocalFields<Basis, 0> f;
    f.u.setZero();
    f.up.setZero();
    const VolumeThetaKernel<Basis, Quad, 0, EmptyTheta> tk(cache, VolumeThetaMapping<Basis, 0>{}, EmptyTheta{});
    const VolumeFKernel<Basis, Quad, 0, EmptyTheta> fk(cache, VolumeFMapping<Basis, 0>{}, EmptyTheta{});
    EXPECT_GT(tk.value(f), 0);
    EXPECT_DOUBLE_EQ(tk.value(f), fk.value(f));
    EXPECT_EQ(tk.gradient(f).size(), 0);
    EXPECT_EQ(tk.hessian(f).size(), 0);
}

TEST(VolumeLocalKernel, PropagatesInvalidFieldsAndInfeasibleStates) {
    const Cache cache{Geometry(coordinates(false))};
    const FK fk(cache, VolumeFMapping<Basis, 2>{}, NeoHookeanEnergyF{});
    const PK pk(cache, VolumeFpMapping<Basis, 2>{}, NeoHookeanEnergyFp{});
    const TK tk(cache, VolumeThetaMapping<Basis, 2>{}, NeoHookeanEnergyTheta{});
    const auto check = [&](const auto& kernel) {
        for (int which = 0; which < 3; ++which) {
            auto bad = fields();
            if (which == 0)
                bad.u(0, 0) = std::numeric_limits<double>::quiet_NaN();
            if (which == 1)
                bad.up(0, 0) = std::numeric_limits<double>::infinity();
            if (which == 2)
                bad.theta(0, 0) = std::numeric_limits<double>::quiet_NaN();
            EXPECT_THROW(kernel.value(bad), std::invalid_argument);
            EXPECT_THROW(kernel.gradient(bad), std::invalid_argument);
            EXPECT_THROW(kernel.hessian(bad), std::invalid_argument);
        }
        auto bad = fields();
        bad.theta.col(1).setConstant(-1);
        EXPECT_THROW(kernel.value(bad), std::invalid_argument);
        EXPECT_THROW(kernel.gradient(bad), std::invalid_argument);
        EXPECT_THROW(kernel.hessian(bad), std::invalid_argument);
        for (bool plastic : {false, true}) {
            bad = fields();
            auto& C = plastic ? bad.up : bad.u;
            C.setZero();
            C.col(0) = -2 * coordinates(false).col(0);
            EXPECT_THROW(kernel.value(bad), InfeasibleState);
            EXPECT_THROW(kernel.gradient(bad), InfeasibleState);
            EXPECT_THROW(kernel.hessian(bad), InfeasibleState);
        }
    };
    check(fk);
    check(pk);
    check(tk);
}

} // namespace
