#include "litepgo/fem/material_inputs.hpp"
#include "litepgo/fem/neo_hookean.hpp"
#include <concepts>
#include <gtest/gtest.h>

static_assert(std::same_as<decltype(litepgo::MaterialInputs::F), Eigen::Matrix3d>);
static_assert(std::same_as<decltype(litepgo::MaterialInputs::Fp), Eigen::Matrix3d>);
static_assert(std::same_as<decltype(litepgo::MaterialInputs::theta), Eigen::VectorXd>);

TEST(MaterialInputs, NeoHookeanRejectsWrongParameterCountsBeforeIndexing) {
    for (int n : {0, 1, 3}) {
        litepgo::MaterialInputs inputs{Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(),
                                       Eigen::VectorXd::Ones(n)};
        EXPECT_THROW(litepgo::NeoHookean{}.value(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean{}.gradient_F(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean{}.hessian_Fp(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean{}.gradient_theta(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean{}.hessian_theta(inputs), std::invalid_argument);
    }
}
