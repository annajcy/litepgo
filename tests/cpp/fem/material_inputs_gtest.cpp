#include "litepgo/fem/material_inputs.hpp"
#include <concepts>
#include <utility>
namespace {
template <int N>
concept CanFormMaterialInputs = requires { typename litepgo::MaterialInputs<N>; };
static_assert(!CanFormMaterialInputs<0>);
static_assert(!CanFormMaterialInputs<-1>);
static_assert(CanFormMaterialInputs<1>);
static_assert(CanFormMaterialInputs<2>);
static_assert(std::same_as<decltype(std::declval<litepgo::MaterialInputs<2>>().F), Eigen::Matrix3d>);
static_assert(std::same_as<decltype(std::declval<litepgo::MaterialInputs<2>>().Fp), Eigen::Matrix3d>);
static_assert(std::same_as<decltype(std::declval<litepgo::MaterialInputs<2>>().theta), Eigen::Vector2d>);
} // namespace
