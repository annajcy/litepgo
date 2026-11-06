#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/dynamic_pattern_response.hpp"
#include "litepgo/energy/response/exp_response.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace litepgo::energy::test {
namespace {
class Affine final : public FixedPatternResponse {
  public:
    explicit Affine(Eigen::MatrixXd A, bool second = true)
        : FixedPatternResponse(A.sparseView(), second ? std::optional<SpMat>(SpMat(A.cols(), A.cols())) : std::nullopt),
          A_(std::move(A)) {}

  private:
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override {
        if (J)
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = A_(it.row(), c);
        return A_ * x + Vec::Ones(num_outputs());
    }
    void eval_hessian_values(const ConstVecRef&, const ConstVecRef&, SpMat&) const override {}
    Eigen::MatrixXd A_;
};
// C2 at zero. Its Jacobian pattern is conservative and fixed; only its Hessian pattern changes.
class Flip final : public DynamicPatternResponse {
  public:
    Flip() : DynamicPatternResponse(Eigen::Matrix2d::Ones().sparseView(), SpMat(2, 2)) {}

  private:
    SpMat pattern_at(const ConstVecRef&) const override {
        return jacobian_pattern();
    }
    SpMat hessian_pattern_at(const ConstVecRef& x) const override {
        SpMat H(2, 2);
        if (x[0] > 0)
            H.insert(0, 0) = 1;
        H.makeCompressed();
        return H;
    }
    Vec eval_values(const ConstVecRef& x, SpMat* J) const override {
        const double s = std::max(0., x[0]);
        if (J) {
            J->coeffRef(0, 0) = -1;
            J->coeffRef(1, 0) = 3 * s * s;
            J->coeffRef(0, 1) = 0;
            J->coeffRef(1, 1) = 1;
        }
        return Eigen::Vector2d(-x[0], x[1] + s * s * s);
    }
    void eval_hessian_values(const ConstVecRef& x, const ConstVecRef& w, SpMat& H) const override {
        if (!linalg::same_pattern(H, hessian_pattern_at(x)))
            throw std::logic_error("call update_pattern");
        if (H.nonZeros())
            H.valuePtr()[0] = 6 * x[0] * w[1];
    }
};
void check_hessian(Response& r, const Vec& x, const Vec& w) {
    r.update_pattern(x);
    const Eigen::MatrixXd H = r.eval_hessian(x, w);
    Eigen::MatrixXd numerical(x.size(), x.size());
    for (int j = 0; j < x.size(); ++j) {
        Vec plus = x, minus = x;
        plus[j] += 1e-6;
        minus[j] -= 1e-6;
        r.update_pattern(plus);
        const Vec gp = r.eval_vjp(plus, w);
        r.update_pattern(minus);
        numerical.col(j) = (gp - r.eval_vjp(minus, w)) / 2e-6;
    }
    EXPECT_LT((H - numerical).norm(), 2e-7);
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
    r.update_pattern(x);
}
} // namespace

TEST(ResponseHessian, AffineAndUnsupportedAreDistinct) {
    Eigen::Matrix<double, 2, 3> A;
    A << .5, .2, .3, 1, -2, 0;
    Affine r(A), old(A, false);
    const Vec x = Vec::Ones(3), w = Eigen::Vector2d(2, -1);
    EXPECT_TRUE(r.has_hessian());
    EXPECT_FALSE(old.has_hessian());
    EXPECT_TRUE(r.has_fixed_pattern());
    EXPECT_FALSE(r.update_pattern(x));
    const SpMat H = r.eval_hessian(x, w);
    EXPECT_EQ(H.rows(), 3);
    EXPECT_EQ(H.cols(), 3);
    EXPECT_EQ(H.nonZeros(), 0);
    EXPECT_THROW(old.hessian_pattern(), std::logic_error);
    EXPECT_THROW(old.eval_hessian(x, w), std::logic_error);
    EXPECT_TRUE(old.eval_jacobian(x).isApprox(r.eval_jacobian(x)));
    check_hessian(r, x, w);
}
TEST(ResponseHessian, ExponentialWeightsDoNotChangeStructure) {
    ExpResponse r(3);
    const Vec x = (Vec(3) << -.4, .2, .7).finished();
    const Vec w = (Vec(3) << 2, 0, -3).finished();
    const Eigen::MatrixXd H = r.eval_hessian(x, w);
    EXPECT_TRUE(H.isApprox(Eigen::MatrixXd(w.cwiseProduct(x.array().exp().matrix()).asDiagonal())));
    const SpMat zero = r.eval_hessian(x, Vec::Zero(3));
    EXPECT_EQ(zero.nonZeros(), 3);
    EXPECT_TRUE(zero.coeffs().isZero());
    EXPECT_TRUE(linalg::same_pattern(zero, r.hessian_pattern()));
    EXPECT_TRUE(r.hessian_pattern().coeffs().isZero());
    check_hessian(r, x, w);
    EXPECT_THROW(r.eval_hessian(Vec::Zero(2), w), std::invalid_argument);
    EXPECT_THROW(r.eval_hessian(x, Vec::Zero(2)), std::invalid_argument);
    EXPECT_THROW(r.eval_hessian(Vec::Constant(3, INFINITY), w), std::invalid_argument);
    EXPECT_THROW(r.eval_hessian(x, Vec::Constant(3, NAN)), std::invalid_argument);
    EXPECT_THROW(r.eval_hessian(Vec::Constant(3, 1000), w), std::domain_error);
    EXPECT_THROW(r.eval_hessian(Vec::Constant(3, -1000), w), std::domain_error);
    EXPECT_THROW(r.eval_hessian(Vec::Ones(3), Vec::Constant(3, std::numeric_limits<double>::max())), std::logic_error);
    EXPECT_NO_THROW(r.eval_hessian(x, w));
}
TEST(ResponseHessian, CompositionIncludesBothNonlinearTerms) {
    auto exp = std::make_shared<ExpResponse>(2);
    ComposedResponse twice(exp, exp);
    const Vec x = Eigen::Vector2d(-.3, .2), w = Eigen::Vector2d(1.3, -.7);
    const Vec e = x.array().exp();
    const Vec expected = w.array() * e.array().exp() * (e.array().square() + e.array());
    EXPECT_TRUE(Eigen::MatrixXd(twice.eval_hessian(x, w)).isApprox(Eigen::MatrixXd(expected.asDiagonal())));
    check_hessian(twice, x, w);
    auto linear = std::make_shared<Affine>((Eigen::Matrix<double, 2, 3>() << .5, .2, .3, -.2, 1, .7).finished());
    ComposedResponse interpolation(exp, linear);
    check_hessian(interpolation, Vec::LinSpaced(3, -.2, .3), w);
    ComposedResponse affine_outer(std::make_shared<Affine>(Eigen::RowVector2d(2, -3)), exp);
    check_hessian(affine_outer, x, Vec::Constant(1, .7));
    ComposedResponse unsupported(exp, std::make_shared<Affine>(Eigen::Matrix2d::Identity(), false));
    EXPECT_FALSE(unsupported.has_hessian());
    EXPECT_THROW(unsupported.eval_hessian(x, w), std::logic_error);
}
TEST(ResponseHessian, DynamicHessianOnlyChangesAndSharedChildRestoration) {
    auto child = std::make_shared<Flip>();
    const Vec x = Eigen::Vector2d(.3, .4), w = Eigen::Vector2d(.2, -.7);
    const SpMat J = child->jacobian_pattern();
    EXPECT_TRUE(child->update_pattern(x));
    EXPECT_TRUE(linalg::same_pattern(J, child->jacobian_pattern()));
    EXPECT_FALSE(child->update_pattern(x));
    EXPECT_TRUE(child->update_pattern(-x));
    EXPECT_EQ(child->hessian_pattern().nonZeros(), 0);
    EXPECT_THROW(child->eval_hessian(x, w), std::logic_error);
    ComposedResponse r(child, child);
    EXPECT_THROW(r.eval_hessian(x, w), std::logic_error);
    for (double sign : {1., -1.}) {
        const Vec point = sign * x;
        EXPECT_TRUE(r.has_hessian());
        r.update_pattern(point);
        child->update_pattern(-point);
        Eigen::Matrix2d expected = Eigen::Matrix2d::Zero();
        expected(0, 0) = w[1] * 6 * std::abs(point[0]);
        EXPECT_TRUE(Eigen::MatrixXd(r.eval_hessian(point, w)).isApprox(expected));
        check_hessian(r, point, w);
    }
}
TEST(ResponseHessian, MixedComponentHessiansAndNestedComposition) {
    class Polynomial final : public FixedPatternResponse {
      public:
        Polynomial()
            : FixedPatternResponse(Eigen::Matrix2d::Ones().sparseView(), Eigen::Matrix2d::Ones().sparseView()) {}

      private:
        Vec eval_values(const ConstVecRef& x, SpMat* J) const override {
            if (J) {
                J->coeffRef(0, 0) = 2 * x[0];
                J->coeffRef(0, 1) = 1;
                J->coeffRef(1, 0) = x[1];
                J->coeffRef(1, 1) = x[0];
            }
            return Eigen::Vector2d(x[0] * x[0] + x[1], x[0] * x[1]);
        }
        void eval_hessian_values(const ConstVecRef&, const ConstVecRef& w, SpMat& H) const override {
            H.coeffRef(0, 0) = 2 * w[0];
            H.coeffRef(0, 1) = H.coeffRef(1, 0) = w[1];
            H.coeffRef(1, 1) = 0;
        }
    };
    const Vec x = Eigen::Vector2d(.3, -.2), w = Eigen::Vector2d(.4, -.7);
    auto polynomial = std::make_shared<Polynomial>();
    const Eigen::Matrix2d expected = (Eigen::Matrix2d() << .8, -.7, -.7, 0).finished();
    EXPECT_TRUE(Eigen::MatrixXd(polynomial->eval_hessian(x, w)).isApprox(expected));
    check_hessian(*polynomial, x, w);
    auto exp = std::make_shared<ExpResponse>(2);
    auto inner = std::make_shared<ComposedResponse>(polynomial, exp);
    ComposedResponse nested(exp, inner);
    check_hessian(nested, x, w);
}
TEST(ResponseHessian, EmptyInputAndOutputSpaces) {
    Affine no_inputs(Eigen::MatrixXd(2, 0)), no_outputs(Eigen::MatrixXd(0, 2));
    EXPECT_EQ(no_inputs.eval_hessian(Vec(0), Vec::Ones(2)).rows(), 0);
    const SpMat H = no_outputs.eval_hessian(Vec::Zero(2), Vec(0));
    EXPECT_EQ(H.rows(), 2);
    EXPECT_EQ(H.nonZeros(), 0);
}
TEST(ResponseHessian, InvalidPatternsAndResultsAreRejected) {
    class Broken final : public FixedPatternResponse {
      public:
        explicit Broken(const SpMat& H) : FixedPatternResponse(Eigen::Matrix2d::Identity().sparseView(), H) {}
        int mode = 0;

      private:
        Vec eval_values(const ConstVecRef& x, SpMat*) const override {
            return x;
        }
        void eval_hessian_values(const ConstVecRef&, const ConstVecRef&, SpMat& H) const override {
            if (mode == 0)
                H.resize(1, 1);
            if (mode == 1)
                H.coeffs().setConstant(NAN);
            if (mode == 2)
                H.coeffRef(0, 1) = 1;
            if (mode == 3)
                H.setZero();
        }
    };
    EXPECT_THROW(Broken{SpMat(1, 1)}, std::invalid_argument);
    SpMat asymmetric(2, 2);
    asymmetric.insert(0, 1) = 1;
    EXPECT_THROW(Broken{asymmetric}, std::invalid_argument);
    Broken r(Eigen::Matrix2d::Ones().sparseView());
    for (int mode : {0, 1, 2, 3}) {
        r.mode = mode;
        EXPECT_THROW(r.eval_hessian(Vec::Zero(2), Vec::Ones(2)), std::logic_error);
    }
}
} // namespace litepgo::energy::test
