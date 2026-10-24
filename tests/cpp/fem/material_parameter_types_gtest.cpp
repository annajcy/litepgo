#include "litepgo/fem/material_parameter_types.hpp"

#include <concepts>

namespace {

template <int N>
concept CanFormMaterialParameterTypes = requires { typename litepgo::MaterialParameterTypes<N>; };

static_assert(CanFormMaterialParameterTypes<0>);
static_assert(CanFormMaterialParameterTypes<1>);
static_assert(CanFormMaterialParameterTypes<2>);
static_assert(!CanFormMaterialParameterTypes<-1>);
static_assert(litepgo::MaterialParameterTypes<2>::num_parameters == 2);
static_assert(std::same_as<litepgo::Mat3, Eigen::Matrix3d>);
static_assert(std::same_as<litepgo::Mat9, Eigen::Matrix<double, 9, 9>>);
static_assert(std::same_as<litepgo::MaterialParameterTypes<0>::Parameters, Eigen::Matrix<double, 0, 1>>);
static_assert(std::same_as<litepgo::MaterialParameterTypes<0>::ParameterHessian, Eigen::Matrix<double, 0, 0>>);
static_assert(std::same_as<litepgo::MaterialParameterTypes<2>::Parameters, Eigen::Vector2d>);
static_assert(std::same_as<litepgo::MaterialParameterTypes<2>::ParameterHessian, Eigen::Matrix2d>);

} // namespace
