#include "litepgo/fem/point_energy.hpp"
#include "litepgo/fem/volumetric_point.hpp"

namespace {

using namespace litepgo;

// A scalar example verifies that the concepts do not prescribe volume inputs.
struct ValueOnly {
    using Point = double;
    double value(const Point&) const;
};
struct FirstOnly : ValueOnly {
    using Gradient = double;
    Gradient gradient(const Point&) const;
};
struct SecondOnly : ValueOnly {
    using Hessian = double;
    Hessian hessian(const Point&) const;
};
struct Both : FirstOnly {
    using Hessian = double;
    Hessian hessian(const Point&) const;
};

static_assert(PointValue<ValueOnly>);
static_assert(!PointFirstOrderEnergy<ValueOnly> && !PointSecondOrderEnergy<ValueOnly>);
static_assert(PointFirstOrderEnergy<FirstOnly> && !PointSecondOrderEnergy<FirstOnly>);
static_assert(!PointFirstOrderEnergy<SecondOnly> && PointSecondOrderEnergy<SecondOnly>);
static_assert(PointFirstOrderEnergy<Both> && PointSecondOrderEnergy<Both>);

struct MissingPoint {};
struct MissingValue : Both {
    double value(const Point&) const = delete;
};
struct WrongValue : Both {
    float value(const Point&) const;
};
struct MutableValue : Both {
    double value(const Point&);
};
struct MutablePoint : Both {
    double value(Point&) const;
};
struct WrongGradient : Both {
    float gradient(const Point&) const;
};
struct WrongHessian : Both {
    float hessian(const Point&) const;
};
struct MutableGradient : Both {
    double gradient(const Point&);
};
struct MutableHessian : Both {
    double hessian(const Point&);
};
struct MissingGradientType : ValueOnly {
    double gradient(const Point&) const;
};
struct MissingHessianType : ValueOnly {
    double hessian(const Point&) const;
};

template <class E>
constexpr bool no_capabilities = !PointValue<E> && !PointFirstOrderEnergy<E> && !PointSecondOrderEnergy<E>;
static_assert(no_capabilities<MissingPoint>);
static_assert(no_capabilities<MissingValue>);
static_assert(no_capabilities<WrongValue>);
static_assert(no_capabilities<MutableValue>);
static_assert(no_capabilities<MutablePoint>);
static_assert(!PointFirstOrderEnergy<WrongGradient> && PointSecondOrderEnergy<WrongGradient>);
static_assert(PointFirstOrderEnergy<WrongHessian> && !PointSecondOrderEnergy<WrongHessian>);
static_assert(!PointFirstOrderEnergy<MutableGradient> && PointSecondOrderEnergy<MutableGradient>);
static_assert(PointFirstOrderEnergy<MutableHessian> && !PointSecondOrderEnergy<MutableHessian>);
static_assert(!PointFirstOrderEnergy<MissingGradientType>);
static_assert(!PointSecondOrderEnergy<MissingHessianType>);

// Zero-parameter volume inputs remain valid, without granting any derivative capability.
template <int N>
concept HasPoint = requires { typename VolumetricPoint<N>; };
static_assert(HasPoint<0> && HasPoint<2> && !HasPoint<-1>);
static_assert(std::same_as<decltype(VolumetricPoint<0>::theta), MaterialParameterTypes<0>::Parameters>);
static_assert(std::same_as<decltype(VolumetricPoint<2>::theta), Eigen::Vector2d>);
struct ZeroParameterValue {
    using Point = VolumetricPoint<0>;
    double value(const Point&) const;
};
static_assert(PointValue<ZeroParameterValue>);
static_assert(!PointFirstOrderEnergy<ZeroParameterValue> && !PointSecondOrderEnergy<ZeroParameterValue>);

} // namespace
