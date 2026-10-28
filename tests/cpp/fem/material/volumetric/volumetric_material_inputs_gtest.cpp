#include "litepgo/fem/material/volumetric/neo_hookean.hpp"
#include "litepgo/fem/material/volumetric/volumetric_material_inputs.hpp"
#include <concepts>
#include <gtest/gtest.h>

static_assert(std::same_as<decltype(litepgo::VolumetricMaterialInputs::F), Eigen::Matrix3d>);
static_assert(std::same_as<decltype(litepgo::VolumetricMaterialInputs::Fp), Eigen::Matrix3d>);
static_assert(std::same_as<decltype(litepgo::VolumetricMaterialInputs::theta), Eigen::VectorXd>);

TEST(VolumetricMaterialInputs, NeoHookeanRejectsWrongParameterCountsBeforeIndexing) {
    for (int n : {0, 1, 3}) {
        litepgo::VolumetricMaterialInputs inputs{Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity(),
                                                 Eigen::VectorXd::Ones(n)};
        EXPECT_THROW(litepgo::NeoHookean::eval_W(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean::eval_dW_dF(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean::eval_d2W_dFp2(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean::eval_dW_dtheta(inputs), std::invalid_argument);
        EXPECT_THROW(litepgo::NeoHookean::eval_d2W_dtheta2(inputs), std::invalid_argument);
    }
}
