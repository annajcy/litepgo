#include "litepgo/dof/constant_dof_map.hpp"
#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/elementwise_response.hpp"
#include "litepgo/energy/response/exp_response.hpp"
#include "litepgo/fem/parameter/log_diagonal_fp_response.hpp"
#include <gtest/gtest.h>

namespace litepgo::energy::test {
TEST(ResponseComposition, SharedAndRepeatedDofsAccumulate) {
    struct Map {
        int num_dofs() const {
            return 2;
        }
        int num_elements() const {
            return 2;
        }
        std::vector<int> dof_indices(int e) const {
            return e == 0 ? std::vector<int>{0, 0} : std::vector<int>{0, 1};
        }
    };
    auto local = std::make_shared<ExpResponse>(2);
    auto element = std::make_shared<ElementwiseResponse>(Map{}, local);
    auto outer = std::make_shared<ExpResponse>(4);
    ComposedResponse response(outer, element);
    const Vec x = Eigen::Vector2d(-.3, .2), v = Eigen::Vector2d(.4, -.7);
    const Vec w = Vec::LinSpaced(4, -.2, .8);
    const SpMat J = response.eval_jacobian(x);
    EXPECT_LT((response.eval_jvp(x, v) - J * v).norm(), 1e-12);
    EXPECT_LT((response.eval_vjp(x, w) - J.transpose() * w).norm(), 1e-12);
    EXPECT_LT((response.eval_jvp(x, v) - (response.eval_value(x + 1e-6 * v) - response.eval_value(x - 1e-6 * v)) / 2e-6)
                  .norm(),
              1e-8);
    const Vec sum = element->eval_vjp(x, Vec::Ones(4));
    EXPECT_NEAR(sum[0], 3 * std::exp(x[0]), 1e-12);
    EXPECT_NEAR(sum[1], std::exp(x[1]), 1e-12);
    EXPECT_THROW(ComposedResponse(nullptr, element), std::invalid_argument);
    EXPECT_THROW(ComposedResponse(local, element), std::invalid_argument);
    EXPECT_THROW(ElementwiseResponse(Map{}, nullptr), std::invalid_argument);
    EXPECT_THROW(ElementwiseResponse(Map{}, std::make_shared<ExpResponse>(3)), std::invalid_argument);
}
TEST(ResponseComposition, ProductsNeverRequestExplicitLocalJacobians) {
    class Direct final : public FixedPatternResponse {
      public:
        Direct() : FixedPatternResponse(Eigen::Matrix2d::Identity().sparseView()) {}

      private:
        Vec eval_impl(const ConstVecRef& x, SpMat* J) const override {
            if (J)
                throw std::logic_error("explicit Jacobian requested");
            return x.array().square();
        }
        Vec eval_jvp_impl(const ConstVecRef& x, const ConstVecRef& v) override {
            return 2 * x.cwiseProduct(v);
        }
        Vec eval_vjp_impl(const ConstVecRef& x, const ConstVecRef& w) override {
            return 2 * x.cwiseProduct(w);
        }
    };
    const mesh::TetTopology topology(4, {{0, 1, 2, 3}, {0, 1, 2, 3}});
    auto direct = std::make_shared<Direct>();
    auto composed = std::make_shared<ComposedResponse>(direct, direct);
    ElementwiseResponse response(dof::ConstantDofMap(topology, 2), composed);
    const Vec x = Eigen::Vector2d(.3, -.4);
    EXPECT_TRUE(response.eval_vjp(x, Vec::Ones(4)).isApprox(8 * x.array().cube().matrix()));
    EXPECT_THROW(response.eval_jacobian(x), std::logic_error);
    EXPECT_TRUE(response.eval_vjp(x, Vec::Ones(4)).isApprox(8 * x.array().cube().matrix()));
}
TEST(ResponseComposition, PlasticTransformAndDomainValidation) {
    for (int n : {2, 3}) {
        fem::LogDiagonalFpResponse response(n);
        const Vec x = Vec::LinSpaced(n, -.2, .3);
        const Vec value = response.eval_value(x);
        const SpMat J = response.eval_jacobian(x);
        EXPECT_TRUE(response.eval_jvp(x, x).isApprox(J * x));
        EXPECT_TRUE(response.eval_vjp(x, value).isApprox(J.transpose() * value));
        EXPECT_THROW(response.eval_value(Vec::Constant(n, -1000)), std::domain_error);
        EXPECT_THROW(response.eval_value(Vec::Constant(n, 1000)), std::domain_error);
        if (n == 2)
            EXPECT_DOUBLE_EQ(value[8], 1.);
        EXPECT_THROW(response.eval_value(Vec::Zero(n + 1)), std::invalid_argument);
    }
    EXPECT_THROW(fem::LogDiagonalFpResponse(1), std::invalid_argument);
    EXPECT_THROW(ExpResponse(0), std::invalid_argument);
}
} // namespace litepgo::energy::test
