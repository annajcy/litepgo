#include "litepgo/energy/response/composed_response.hpp"
#include "litepgo/energy/response/dynamic_pattern_response.hpp"
#include "litepgo/energy/response/exp_response.hpp"
#include "litepgo/linalg/sparse_pattern.hpp"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace litepgo::energy::test {
namespace {
class Affine final : public FixedPatternResponse {
  public:
    explicit Affine(Eigen::MatrixXd A, bool second = true)
        : FixedPatternResponse(A.sparseView(), second ? std::optional<SpMat>(SpMat(A.cols(), A.cols())) : std::nullopt),
          A_(std::move(A)) {}

  private:
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        if (J)
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = A_(it.row(), c);
        return A_ * x + Vec::Ones(num_outputs());
    }
    void eval_hessian_impl(const ConstVecRef&, const ConstVecRef&, SpMat&) override {}
    Eigen::MatrixXd A_;
};
// C2 at zero. Its Jacobian pattern is conservative and fixed; only its Hessian pattern changes.
class Flip final : public DynamicPatternResponse {
  public:
    Flip() : DynamicPatternResponse(2, 2) {}
    bool has_hessian() const override {
        return true;
    }

  private:
    SpMat jacobian_pattern_at(const ConstVecRef&) const override {
        return Eigen::Matrix2d::Ones().sparseView();
    }
    SpMat hessian_pattern_at(const ConstVecRef& x) const override {
        SpMat H(2, 2);
        if (x[0] > 0)
            H.insert(0, 0) = 1;
        H.makeCompressed();
        return H;
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        const double s = std::max(0., x[0]);
        if (J) {
            J->coeffRef(0, 0) = -1;
            J->coeffRef(1, 0) = 3 * s * s;
            J->coeffRef(0, 1) = 0;
            J->coeffRef(1, 1) = 1;
        }
        return Eigen::Vector2d(-x[0], x[1] + s * s * s);
    }
    void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& w, SpMat& H) override {
        if (!linalg::same_pattern(H, hessian_pattern_at(x)))
            throw std::logic_error("Incorrect Hessian structure");
        if (H.nonZeros())
            H.valuePtr()[0] = 6 * x[0] * w[1];
    }
};
void check_hessian(Response& r, const Vec& x, const Vec& w) {
    const Eigen::MatrixXd H = r.evaluate(x)->hessian(w);
    Eigen::MatrixXd numerical(x.size(), x.size());
    for (int j = 0; j < x.size(); ++j) {
        Vec plus = x, minus = x;
        plus[j] += 1e-6;
        minus[j] -= 1e-6;
        const Vec gp = r.evaluate(plus)->vjp(w);
        numerical.col(j) = (gp - r.evaluate(minus)->vjp(w)) / 2e-6;
    }
    EXPECT_LT((H - numerical).norm(), 2e-7);
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
}
} // namespace

TEST(ResponseHessian, AffineAndUnsupportedAreDistinct) {
    Eigen::Matrix<double, 2, 3> A;
    A << .5, .2, .3, 1, -2, 0;
    Affine r(A), old(A, false);
    const Vec x = Vec::Ones(3), w = Eigen::Vector2d(2, -1);
    EXPECT_TRUE(r.has_hessian());
    EXPECT_FALSE(old.has_hessian());
    const SpMat H = r.evaluate(x)->hessian(w);
    EXPECT_EQ(H.rows(), 3);
    EXPECT_EQ(H.cols(), 3);
    EXPECT_EQ(H.nonZeros(), 0);
    EXPECT_THROW(old.evaluate(x)->hessian(w), std::logic_error);
    EXPECT_TRUE(old.evaluate(x)->jacobian().isApprox(r.evaluate(x)->jacobian()));
    check_hessian(r, x, w);
}
TEST(ResponseHessian, ExponentialWeightsDoNotChangeStructure) {
    ExpResponse r(3);
    const Vec x = (Vec(3) << -.4, .2, .7).finished();
    const Vec w = (Vec(3) << 2, 0, -3).finished();
    const Eigen::MatrixXd H = r.evaluate(x)->hessian(w);
    EXPECT_TRUE(H.isApprox(Eigen::MatrixXd(w.cwiseProduct(x.array().exp().matrix()).asDiagonal())));
    const SpMat zero = r.evaluate(x)->hessian(Vec::Zero(3));
    EXPECT_EQ(zero.nonZeros(), 3);
    EXPECT_TRUE(zero.coeffs().isZero());
    check_hessian(r, x, w);
    EXPECT_THROW(r.evaluate(Vec::Zero(2))->hessian(w), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->hessian(Vec::Zero(2)), std::invalid_argument);
    EXPECT_THROW(r.evaluate(Vec::Constant(3, INFINITY))->hessian(w), std::invalid_argument);
    EXPECT_THROW(r.evaluate(x)->hessian(Vec::Constant(3, NAN)), std::invalid_argument);
    EXPECT_THROW(r.evaluate(Vec::Constant(3, 1000))->hessian(w), std::domain_error);
    EXPECT_THROW(r.evaluate(Vec::Constant(3, -1000))->hessian(w), std::domain_error);
    EXPECT_THROW(r.evaluate(Vec::Ones(3))->hessian(Vec::Constant(3, std::numeric_limits<double>::max())),
                 std::logic_error);
    EXPECT_NO_THROW(r.evaluate(x)->hessian(w));
}
TEST(ResponseHessian, CompositionIncludesBothNonlinearTerms) {
    auto exp = std::make_shared<ExpResponse>(2);
    ComposedResponse twice(exp, exp);
    const Vec x = Eigen::Vector2d(-.3, .2), w = Eigen::Vector2d(1.3, -.7);
    const Vec e = x.array().exp();
    const Vec expected = w.array() * e.array().exp() * (e.array().square() + e.array());
    EXPECT_TRUE(Eigen::MatrixXd(twice.evaluate(x)->hessian(w)).isApprox(Eigen::MatrixXd(expected.asDiagonal())));
    check_hessian(twice, x, w);
    auto linear = std::make_shared<Affine>((Eigen::Matrix<double, 2, 3>() << .5, .2, .3, -.2, 1, .7).finished());
    ComposedResponse interpolation(exp, linear);
    check_hessian(interpolation, Vec::LinSpaced(3, -.2, .3), w);
    ComposedResponse affine_outer(std::make_shared<Affine>(Eigen::RowVector2d(2, -3)), exp);
    check_hessian(affine_outer, x, Vec::Constant(1, .7));
    ComposedResponse unsupported(exp, std::make_shared<Affine>(Eigen::Matrix2d::Identity(), false));
    EXPECT_FALSE(unsupported.has_hessian());
    EXPECT_THROW(unsupported.evaluate(x)->hessian(w), std::logic_error);
}
TEST(ResponseHessian, DynamicHessianPatternsAndSharedChildEvaluation) {
    auto child = std::make_shared<Flip>();
    const Vec x = Eigen::Vector2d(.3, .4), w = Eigen::Vector2d(.2, -.7);
    ComposedResponse r(child, child);
    for (double sign : {1., -1.}) {
        const Vec point = sign * x;
        EXPECT_TRUE(r.has_hessian());
        Eigen::Matrix2d expected = Eigen::Matrix2d::Zero();
        expected(0, 0) = w[1] * 6 * std::abs(point[0]);
        EXPECT_TRUE(Eigen::MatrixXd(r.evaluate(point)->hessian(w)).isApprox(expected));
        check_hessian(r, point, w);
    }
}
TEST(ResponseHessian, MixedComponentHessiansAndNestedComposition) {
    class Polynomial final : public FixedPatternResponse {
      public:
        Polynomial()
            : FixedPatternResponse(Eigen::Matrix2d::Ones().sparseView(), Eigen::Matrix2d::Ones().sparseView()) {}

      private:
        Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
            if (J) {
                J->coeffRef(0, 0) = 2 * x[0];
                J->coeffRef(0, 1) = 1;
                J->coeffRef(1, 0) = x[1];
                J->coeffRef(1, 1) = x[0];
            }
            return Eigen::Vector2d(x[0] * x[0] + x[1], x[0] * x[1]);
        }
        void eval_hessian_impl(const ConstVecRef&, const ConstVecRef& w, SpMat& H) override {
            H.coeffRef(0, 0) = 2 * w[0];
            H.coeffRef(0, 1) = H.coeffRef(1, 0) = w[1];
            H.coeffRef(1, 1) = 0;
        }
    };
    const Vec x = Eigen::Vector2d(.3, -.2), w = Eigen::Vector2d(.4, -.7);
    auto polynomial = std::make_shared<Polynomial>();
    const Eigen::Matrix2d expected = (Eigen::Matrix2d() << .8, -.7, -.7, 0).finished();
    EXPECT_TRUE(Eigen::MatrixXd(polynomial->evaluate(x)->hessian(w)).isApprox(expected));
    check_hessian(*polynomial, x, w);
    auto exp = std::make_shared<ExpResponse>(2);
    auto inner = std::make_shared<ComposedResponse>(polynomial, exp);
    ComposedResponse nested(exp, inner);
    check_hessian(nested, x, w);
}
TEST(ResponseHessian, EmptyInputAndOutputSpaces) {
    Affine no_inputs(Eigen::MatrixXd(2, 0)), no_outputs(Eigen::MatrixXd(0, 2));
    EXPECT_EQ(no_inputs.evaluate(Vec(0))->hessian(Vec::Ones(2)).rows(), 0);
    const SpMat H = no_outputs.evaluate(Vec::Zero(2))->hessian(Vec(0));
    EXPECT_EQ(H.rows(), 2);
    EXPECT_EQ(H.nonZeros(), 0);
}
TEST(ResponseHessian, InvalidPatternsAndResultsAreRejected) {
    class Broken final : public FixedPatternResponse {
      public:
        explicit Broken(const SpMat& H) : FixedPatternResponse(Eigen::Matrix2d::Identity().sparseView(), H) {}
        int mode = 0;

      private:
        Vec eval_impl(const ConstVecRef& x, SpMat*) override {
            return x;
        }
        void eval_hessian_impl(const ConstVecRef&, const ConstVecRef&, SpMat& H) override {
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
#ifndef NDEBUG
    const std::array modes{0, 1, 2, 3};
#else
    const std::array modes{0, 1};
#endif
    for (int mode : modes) {
        r.mode = mode;
        EXPECT_THROW(r.evaluate(Vec::Zero(2))->hessian(Vec::Ones(2)), std::logic_error);
    }
}
} // namespace litepgo::energy::test

namespace litepgo::energy::test {
namespace {
class PositiveCube final : public DynamicPatternResponse {
  public:
    PositiveCube() : DynamicPatternResponse(1, 1) {}
    bool has_hessian() const override {
        return true;
    }
    mutable int jacobian_queries = 0, hessian_queries = 0;
    bool invalid_hessian = false;

  private:
    static SpMat pattern(const ConstVecRef& x) {
        SpMat p(1, 1);
        if (x[0] > 0)
            p.insert(0, 0) = 1;
        p.makeCompressed();
        return p;
    }
    SpMat jacobian_pattern_at(const ConstVecRef& x) const override {
        ++jacobian_queries;
        return pattern(x);
    }
    SpMat hessian_pattern_at(const ConstVecRef& x) const override {
        ++hessian_queries;
        return invalid_hessian ? SpMat(2, 2) : pattern(x);
    }
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        const double s = std::max(0., x[0]);
        if (J && J->nonZeros())
            J->valuePtr()[0] = 3 * s * s;
        return Vec::Constant(1, s * s * s);
    }
    void eval_hessian_impl(const ConstVecRef& x, const ConstVecRef& w, SpMat& H) override {
        if (H.nonZeros())
            H.valuePtr()[0] = 6 * x[0] * w[0];
    }
};
} // namespace

TEST(ResponseEvaluation, IndependentPatternsAndRecovery) {
    PositiveCube r;
    const Vec x = Vec::Ones(1), w = Vec::Constant(1, 2);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->value()[0], 1);
    EXPECT_EQ(r.jacobian_queries + r.hessian_queries, 0);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->jacobian().coeff(0, 0), 3);
    EXPECT_EQ(r.hessian_queries, 0);
    const int queries = r.jacobian_queries;
    EXPECT_DOUBLE_EQ(r.evaluate(x)->hessian(w).coeff(0, 0), 12);
    EXPECT_EQ(r.jacobian_queries, queries);
    const SpMat saved = r.evaluate(x)->jacobian();
    EXPECT_EQ(r.evaluate(-x)->jacobian().nonZeros(), 0);
    EXPECT_DOUBLE_EQ(saved.coeff(0, 0), 3);
    r.invalid_hessian = true;
    EXPECT_THROW(r.evaluate(x)->hessian(w), std::invalid_argument);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->jacobian().coeff(0, 0), 3);
    r.invalid_hessian = false;
    EXPECT_DOUBLE_EQ(r.evaluate(x)->hessian(w).coeff(0, 0), 12);
    EXPECT_EQ(r.evaluate(-x)->hessian(w).nonZeros(), 0);
}
} // namespace litepgo::energy::test

namespace litepgo::energy::test {
TEST(ResponseEvaluation, MathematicalSubclassNeedsNoSparseProtocolAndAllowsValueReentry) {
    class Map final : public Response {
      public:
        int num_inputs() const noexcept override {
            return 1;
        }
        int num_outputs() const noexcept override {
            return 1;
        }
        bool has_hessian() const override {
            return true;
        }
        bool bad_shape = false;

      private:
        Vec value_at(const ConstVecRef& x) override {
            return x.array().square();
        }
        std::pair<Vec, SpMat> value_jacobian_at(const ConstVecRef& x) override {
            SpMat J(bad_shape ? 2 : 1, 1);
            J.insert(0, 0) = 2 * x[0];
            J.makeCompressed();
            return {value_at(x), J};
        }
        SpMat hessian_at(const ConstVecRef&, const ConstVecRef& weights) override {
            return Eigen::MatrixXd::Constant(1, 1, 2 * weights[0]).sparseView();
        }
    } r;
    const Vec x = Vec::Constant(1, 3), w = Vec::Constant(1, 2);
    auto evaluation_0 = r.evaluate(x);
    auto jacobian_0 = evaluation_0->jacobian();
    auto [value, J] = std::pair{litepgo::Vec(evaluation_0->value()), std::move(jacobian_0)};
    EXPECT_DOUBLE_EQ(value[0], 9);
    EXPECT_DOUBLE_EQ(J.coeff(0, 0), 6);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->jvp(w)[0], 12);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->vjp(w)[0], 12);
    EXPECT_DOUBLE_EQ(r.evaluate(x)->hessian(w).coeff(0, 0), 4);
    r.bad_shape = true;
    EXPECT_THROW(r.evaluate(x)->jacobian(), std::logic_error);
    r.bad_shape = false;
    EXPECT_DOUBLE_EQ(r.evaluate(x)->jacobian().coeff(0, 0), 6);
}
} // namespace litepgo::energy::test

namespace litepgo::energy::test {
TEST(ResponseEvaluation, PatternNormalizationReusesOwnedStorageAndPreservesZeroSlots) {
    struct PatternAccess : detail::SparseResponse {
        using SparseResponse::checked_pattern;
    };
    SpMat input(2, 2);
    input.insert(0, 0) = 7;
    input.insert(1, 1) = 0;
    input.makeCompressed();
    const double* values = input.valuePtr();
    const int* indices = input.innerIndexPtr();
    SpMat normalized = PatternAccess::checked_pattern(std::move(input), 2, 2, true);
    EXPECT_EQ(normalized.valuePtr(), values);
    EXPECT_EQ(normalized.innerIndexPtr(), indices);
    EXPECT_EQ(normalized.nonZeros(), 2);
    EXPECT_TRUE(normalized.coeffs().isZero());
    normalized.coeffs().setConstant(3);
    const SpMat snapshot = PatternAccess::checked_pattern(normalized, 2, 2, true);
    EXPECT_TRUE(snapshot.coeffs().isZero());
    EXPECT_TRUE(normalized.coeffs().isConstant(3));
    EXPECT_THROW(PatternAccess::checked_pattern(normalized, 3, 3), std::invalid_argument);
    SpMat asymmetric(2, 2);
    asymmetric.insert(0, 1) = 1;
    EXPECT_THROW(PatternAccess::checked_pattern(std::move(asymmetric), 2, 2, true), std::invalid_argument);
}
} // namespace litepgo::energy::test

#ifndef NDEBUG
namespace litepgo::energy::test {
TEST(ResponseEvaluation, DebugRejectsNoncanonicalOwningResults) {
    class Uncompressed final : public Response {
      public:
        int num_inputs() const noexcept override {
            return 2;
        }
        int num_outputs() const noexcept override {
            return 2;
        }
        bool has_hessian() const override {
            return true;
        }

      private:
        Vec value_at(const ConstVecRef& x) override {
            return x;
        }
        std::pair<Vec, SpMat> value_jacobian_at(const ConstVecRef& x) override {
            return {x, matrix()};
        }
        SpMat hessian_at(const ConstVecRef&, const ConstVecRef&) override {
            return matrix();
        }
        static SpMat matrix() {
            SpMat result(2, 2);
            result.insert(0, 0) = 1;
            result.insert(1, 1) = 1;
            EXPECT_FALSE(result.isCompressed());
            return result;
        }
    } r;
    EXPECT_THROW(r.evaluate(Vec::Ones(2))->jacobian(), std::logic_error);
    EXPECT_THROW(r.evaluate(Vec::Ones(2))->hessian(Vec::Ones(2)), std::logic_error);
}
} // namespace litepgo::energy::test
#endif
