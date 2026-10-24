#include "litepgo/fem/local_kernel.hpp"

#include <array>
#include <gtest/gtest.h>

namespace {

// No volume input, basis, or FEM field types: a two-component affine function
// of three local coefficients, sampled with a small custom quadrature cache.
struct AffineMapping {
    using Fields = Eigen::Vector3d;
    using ReferencePoint = double;
    using Point = Eigen::Vector2d;
    using PointGradient = Eigen::Vector2d;
    using PointHessian = Eigen::Matrix2d;
    using Jacobian = Eigen::Matrix<double, 2, 3>;
    static constexpr int num_dofs = 3;
    static constexpr int point_dofs = 2;
    double scale;

    Jacobian active_jacobian(const ReferencePoint& xi) const {
        Jacobian B;
        B << 1, xi, 2, -xi, 1, 3;
        return scale * B;
    }
    Point point_input(const Fields& fields, const ReferencePoint& xi) const {
        return active_jacobian(xi) * fields + Point(xi, 0.5);
    }
};
struct Cache {
    struct Sample {
        double point;
        double volume_weight;
    };
    static constexpr int num_quadrature_points = 2;
    std::array<Sample, 2> samples{{{0.2, 0.4}, {0.8, 0.6}}};
    const Sample& sample(int q) const {
        return samples.at(static_cast<std::size_t>(q));
    }
};
struct Energy {
    using Point = Eigen::Vector2d;
    using Gradient = Eigen::Vector2d;
    using Hessian = Eigen::Matrix2d;
    double value(const Point& p) const {
        return 0.5 * p.squaredNorm();
    }
    Gradient gradient(const Point& p) const {
        return p;
    }
    Hessian hessian(const Point&) const {
        return Hessian::Identity();
    }
};
using Kernel = litepgo::LocalKernel<AffineMapping, Cache, Energy>;

struct WrongPointEnergy : Energy {
    using Point = double;
    double value(const Point&) const;
};
struct WrongReferenceCache : Cache {
    struct Sample {
        Eigen::Vector2d point;
        double volume_weight;
    };
    const Sample& sample(int) const;
};
template <class C, class E>
concept CanFormKernel = requires { typename litepgo::LocalKernel<AffineMapping, C, E>; };
static_assert(CanFormKernel<Cache, Energy>);
static_assert(!CanFormKernel<Cache, WrongPointEnergy>);
static_assert(!CanFormKernel<WrongReferenceCache, Energy>);

TEST(LocalKernel, GenericAffineMappingAndOwnedState) {
    Cache cache;
    AffineMapping mapping{1.3};
    const Kernel kernel(cache, mapping, Energy{});
    const Eigen::Vector3d fields(0.2, -0.4, 0.6);
    double expected = 0;
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (const auto& sample : cache.samples) {
        const auto B = mapping.active_jacobian(sample.point);
        const auto point = mapping.point_input(fields, sample.point);
        expected += sample.volume_weight * 0.5 * point.squaredNorm();
        g += sample.volume_weight * B.transpose() * point;
        H += sample.volume_weight * B.transpose() * B;
    }
    cache.samples[0].volume_weight = 100;
    mapping.scale = 99;
    EXPECT_NEAR(kernel.value(fields), expected, 1e-13);
    EXPECT_TRUE(kernel.gradient(fields).isApprox(g, 1e-13));
    EXPECT_TRUE(kernel.hessian(fields).isApprox(H, 1e-13));
    const double h = 1e-4;
    for (int j = 0; j < 3; ++j) {
        auto plus = fields, minus = fields;
        plus[j] += h;
        minus[j] -= h;
        EXPECT_NEAR(g[j], (kernel.value(plus) - kernel.value(minus)) / (2 * h), 1e-10);
        EXPECT_TRUE(H.col(j).isApprox((kernel.gradient(plus) - kernel.gradient(minus)) / (2 * h), 1e-10));
    }
}

} // namespace
