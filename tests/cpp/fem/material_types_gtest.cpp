#include "litepgo/fem/material_types.h"

#include <concepts>

namespace {

template <int N>
concept CanFormMaterialTypes = requires { typename litepgo::MaterialTypes<N>; };

static_assert(CanFormMaterialTypes<0>);
static_assert(CanFormMaterialTypes<1>);
static_assert(CanFormMaterialTypes<2>);
static_assert(!CanFormMaterialTypes<-1>);
static_assert(litepgo::MaterialTypes<2>::num_parameters == 2);
static_assert(std::same_as<litepgo::Mat3, Eigen::Matrix3d>);
static_assert(std::same_as<litepgo::Mat9, Eigen::Matrix<double, 9, 9>>);
static_assert(std::same_as<litepgo::MaterialTypes<0>::Parameters, Eigen::Matrix<double, 0, 1>>);
static_assert(std::same_as<litepgo::MaterialTypes<0>::ParameterHessian, Eigen::Matrix<double, 0, 0>>);
static_assert(std::same_as<litepgo::MaterialTypes<2>::Parameters, Eigen::Vector2d>);
static_assert(std::same_as<litepgo::MaterialTypes<2>::ParameterHessian, Eigen::Matrix2d>);

} // namespace
