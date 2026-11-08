#include "litepgo/fem/quadrature/1d/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/3d/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"

namespace litepgo::fem::test {

namespace {

struct OnePointRule {
    static constexpr int num_points = 1;
    static const Eigen::Matrix<double, 1, 1>& points();
    static const Eigen::Matrix<double, 1, 1>& weights();
};

struct MissingCount {};

struct MissingPoints {
    static constexpr int num_points = 1;
    static const Eigen::Matrix<double, 1, 1>& weights();
};

struct MissingWeights {
    static constexpr int num_points = 1;
    static const Eigen::Matrix<double, 1, 1>& points();
};

struct NonConstantCount : OnePointRule {
    inline static int num_points = 1;
};

struct ZeroCount : OnePointRule {
    static constexpr int num_points = 0;
};

struct NegativeCount : OnePointRule {
    static constexpr int num_points = -1;
};

struct WrongPointRows : OnePointRule {
    static const Eigen::Matrix<double, 2, 1>& points();
};

struct WrongWeightRows : OnePointRule {
    static const Eigen::Matrix<double, 2, 1>& weights();
};

struct WrongWeightColumns : OnePointRule {
    static const Eigen::Matrix<double, 1, 2>& weights();
};

struct WrongScalar : OnePointRule {
    static const Eigen::Matrix<float, 1, 1>& points();
};

struct PointsByValue : OnePointRule {
    static Eigen::Matrix<double, 1, 1> points();
};

struct MutableWeights : OnePointRule {
    static Eigen::Matrix<double, 1, 1>& weights();
};

struct TwoDimensionalRule : OnePointRule {
    static const Eigen::Matrix<double, 1, 2>& points();
};

static_assert(litepgo::fem::Quadrature1D<OnePointRule>);
static_assert(litepgo::fem::Quadrature1D<litepgo::fem::GaussLegendre1D<2>>);
static_assert(litepgo::fem::Quadrature3D<litepgo::fem::GaussLegendre3D<2>>);
static_assert(litepgo::fem::Quadrature<TwoDimensionalRule, 2>);
static_assert(!litepgo::fem::Quadrature1D<MissingCount>);
static_assert(!litepgo::fem::Quadrature1D<MissingPoints>);
static_assert(!litepgo::fem::Quadrature1D<MissingWeights>);
static_assert(!litepgo::fem::Quadrature1D<NonConstantCount>);
static_assert(!litepgo::fem::Quadrature1D<ZeroCount>);
static_assert(!litepgo::fem::Quadrature1D<NegativeCount>);
static_assert(!litepgo::fem::Quadrature1D<WrongPointRows>);
static_assert(!litepgo::fem::Quadrature1D<WrongWeightRows>);
static_assert(!litepgo::fem::Quadrature1D<WrongWeightColumns>);
static_assert(!litepgo::fem::Quadrature1D<WrongScalar>);
static_assert(!litepgo::fem::Quadrature1D<PointsByValue>);
static_assert(!litepgo::fem::Quadrature1D<MutableWeights>);
static_assert(!litepgo::fem::Quadrature1D<TwoDimensionalRule>);
static_assert(!litepgo::fem::Quadrature3D<OnePointRule>);
static_assert(!litepgo::fem::Quadrature1D<litepgo::fem::GaussLegendre3D<2>>);
static_assert(!litepgo::fem::Quadrature3D<litepgo::fem::GaussLegendre1D<2>>);
static_assert(!litepgo::fem::Quadrature<OnePointRule, 0>);
static_assert(!litepgo::fem::Quadrature<OnePointRule, -1>);

} // namespace

} // namespace litepgo::fem::test
