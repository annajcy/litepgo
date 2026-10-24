#include "litepgo/fem/quadrature/gauss_legendre_1d.hpp"
#include "litepgo/fem/quadrature/gauss_legendre_3d.hpp"
#include "litepgo/fem/quadrature/quadrature.hpp"
#include "litepgo/fem/quadrature/tensor_product_quadrature.hpp"

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

template <class Q>
concept CanFormTensorProduct = requires { typename litepgo::TensorProductQuadrature3D<Q>; };

static_assert(litepgo::Quadrature1D<OnePointRule>);
static_assert(litepgo::Quadrature1D<litepgo::GaussLegendre1D<2>>);
static_assert(litepgo::Quadrature3D<litepgo::GaussLegendre3D<2>>);
static_assert(litepgo::Quadrature3D<litepgo::TensorProductQuadrature3D<OnePointRule>>);
static_assert(litepgo::Quadrature<TwoDimensionalRule, 2>);
static_assert(!litepgo::Quadrature1D<MissingCount>);
static_assert(!litepgo::Quadrature1D<MissingPoints>);
static_assert(!litepgo::Quadrature1D<MissingWeights>);
static_assert(!litepgo::Quadrature1D<NonConstantCount>);
static_assert(!litepgo::Quadrature1D<ZeroCount>);
static_assert(!litepgo::Quadrature1D<NegativeCount>);
static_assert(!litepgo::Quadrature1D<WrongPointRows>);
static_assert(!litepgo::Quadrature1D<WrongWeightRows>);
static_assert(!litepgo::Quadrature1D<WrongWeightColumns>);
static_assert(!litepgo::Quadrature1D<WrongScalar>);
static_assert(!litepgo::Quadrature1D<PointsByValue>);
static_assert(!litepgo::Quadrature1D<MutableWeights>);
static_assert(!litepgo::Quadrature1D<TwoDimensionalRule>);
static_assert(!litepgo::Quadrature3D<OnePointRule>);
static_assert(!litepgo::Quadrature1D<litepgo::GaussLegendre3D<2>>);
static_assert(!litepgo::Quadrature3D<litepgo::GaussLegendre1D<2>>);
static_assert(!litepgo::Quadrature<OnePointRule, 0>);
static_assert(!litepgo::Quadrature<OnePointRule, -1>);
static_assert(CanFormTensorProduct<OnePointRule>);
static_assert(CanFormTensorProduct<litepgo::GaussLegendre1D<2>>);
static_assert(!CanFormTensorProduct<WrongScalar>);
static_assert(!CanFormTensorProduct<MissingWeights>);
static_assert(!CanFormTensorProduct<litepgo::GaussLegendre3D<2>>);

} // namespace
