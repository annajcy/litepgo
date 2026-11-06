#include "litepgo/energy/response/dynamic_pattern_response.hpp"
#include "litepgo/energy/response/fixed_pattern_response.hpp"
#include "litepgo/linalg/sparse_solver/eigen_ldlt.hpp"
#include "litepgo/solver/gauss_newton.hpp"
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

using namespace litepgo;
using namespace litepgo::solver;

class TestResponse final : public energy::FixedPatternResponse {
  public:
    using Evaluator = std::function<Vec(const ConstVecRef&, SpMat*)>;
    TestResponse(const SpMat& pattern, Evaluator f) : FixedPatternResponse(pattern), evaluator(std::move(f)) {}

  private:
    Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
        return evaluator(x, J);
    }
    Evaluator evaluator;
};
static_assert(std::is_abstract_v<energy::FixedPatternResponse>);

// Test adapter keeps the small analytic fixtures readable; the solver receives an object.
template <class F> auto make_response(int rows, int cols, F evaluator) {
    SpMat pattern = Eigen::MatrixXd::Ones(rows, cols).sparseView();
    return TestResponse(pattern, [evaluator](const ConstVecRef& x, SpMat* J) -> Vec {
        Eigen::MatrixXd dense;
        Vec r = evaluator(x, J ? &dense : nullptr);
        if (J) {
            if (dense.rows() != J->rows() || dense.cols() != J->cols())
                throw std::logic_error("Invalid test Jacobian");
            for (int c = 0; c < J->outerSize(); ++c)
                for (SpMat::InnerIterator it(*J, c); it; ++it)
                    it.valueRef() = dense(it.row(), c);
        }
        return r;
    });
}

TEST(GaussNewton, OverdeterminedNonzeroResidual) {
    int values = 0, derivatives = 0;
    auto f = make_response(3, 2, [&](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (J) {
            ++derivatives;
            *J = Eigen::Matrix<double, 3, 2>{{1, 0}, {0, 1}, {1, 0}};
        } else
            ++values;
        return (Vec(3) << x[0] - 1, x[1] - 2, x[0] - 3).finished();
    });
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-8;
    const auto result = gauss_newton_solve(f, Vec::Zero(2), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x[0], 2, 1e-8);
    EXPECT_NEAR(result.x[1], 2, 1e-8);
    EXPECT_NEAR(result.objective, 1, 1e-12);
    EXPECT_GT(values, 0);
    EXPECT_EQ(result.evaluations, values + derivatives);
}

TEST(GaussNewton, NonlinearAndInfeasibleTrials) {
    int rejected = 0;
    auto f = make_response(1, 1, [&](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (x[0] <= 0) {
            ++rejected;
            throw std::domain_error("positive only");
        }
        if (J)
            *J = Eigen::MatrixXd::Constant(1, 1, 1 / x[0]);
        return Vec::Constant(1, std::log(x[0]));
    });
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    const auto result = gauss_newton_solve(f, Vec::Constant(1, 10), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x[0], 1, 1e-8);
    EXPECT_GT(rejected, 0);
    EXPECT_THROW(gauss_newton_solve(f, Vec::Constant(1, -1), options), std::domain_error);
}

TEST(GaussNewton, LimitsAndValidation) {
    GaussNewtonOptions options;
    auto f = make_response(1, 1, [](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Identity(1, 1);
        return x;
    });
    EXPECT_THROW(gauss_newton_solve(f, Vec::Ones(1), options), std::invalid_argument);
    options.gradient_tolerance = 1e-8;
    options.max_iterations = 0;
    EXPECT_EQ(gauss_newton_solve(f, Vec::Ones(1), options).status, GaussNewtonStatus::MaxIterations);
    options.max_iterations = 50;
    auto blocked = make_response(1, 1, [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (!J)
            throw std::domain_error("trial blocked");
        *J = Eigen::MatrixXd::Ones(1, 1);
        return Vec::Ones(1);
    });
    EXPECT_EQ(gauss_newton_solve(blocked, Vec::Zero(1), options).status, GaussNewtonStatus::StepTooSmall);
    auto bad = make_response(1, 1, [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Zero(2, 2);
        return Vec::Ones(1);
    });
    EXPECT_THROW(gauss_newton_solve(bad, Vec::Zero(1), options), std::logic_error);
    auto nan = make_response(1, 1, [](const ConstVecRef&, Eigen::MatrixXd*) -> Vec {
        return Vec::Constant(1, std::numeric_limits<double>::quiet_NaN());
    });
    EXPECT_THROW(gauss_newton_solve(nan, Vec::Zero(1), options), std::logic_error);
}

TEST(GaussNewton, RankDeficientAndCallbackErrors) {
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-8;
    auto rank_one = make_response(1, 2, [](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Ones(1, 2);
        return Vec::Constant(1, x.sum() - 2);
    });
    const auto result = gauss_newton_solve(rank_one, Vec::Zero(2), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x.sum(), 2, 1e-8);
    auto changed_size = make_response(1, 1, [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::MatrixXd::Ones(1, 1);
        return Vec::Ones(J ? 1 : 2);
    });
    EXPECT_THROW(gauss_newton_solve(changed_size, Vec::Zero(1), options), std::logic_error);
    auto broken_trial = make_response(1, 1, [](const ConstVecRef&, Eigen::MatrixXd* J) -> Vec {
        if (!J)
            throw std::runtime_error("broken callback");
        *J = Eigen::MatrixXd::Ones(1, 1);
        return Vec::Ones(1);
    });
    EXPECT_THROW(gauss_newton_solve(broken_trial, Vec::Zero(1), options), std::runtime_error);
}

TEST(Response, PatternAndInputContracts) {
    SpMat pattern(2, 1);
    pattern.insert(0, 0) = 1;
    TestResponse r(pattern, [](const ConstVecRef& x, SpMat* J) -> Vec {
        if (J)
            J->coeffRef(0, 0) = 2;
        return (Vec(2) << 2 * x[0], 0).finished();
    });
    EXPECT_THROW(r.eval_value(Vec::Zero(2)), std::invalid_argument);
    SpMat wrong(2, 1);
    EXPECT_EQ(r.eval_jacobian(Vec::Zero(1)).coeff(0, 0), 2);
    TestResponse broken(pattern, [](const ConstVecRef&, SpMat* J) -> Vec {
        if (J)
            J->coeffRef(1, 0) = 1;
        return Vec::Zero(2);
    });
#ifndef NDEBUG
    EXPECT_THROW(broken.eval_jacobian(Vec::Zero(1)), std::logic_error);
#endif
}

TEST(Response, DynamicPatternRefreshesAtAcceptedStates) {
    class Changing final : public energy::DynamicPatternResponse {
      public:
        Changing() : DynamicPatternResponse(1, 1) {}

      private:
        SpMat jacobian_pattern_at(const ConstVecRef&) const override {
            SpMat p(1, 1);
            p.insert(0, 0) = 0;
            return p;
        }
        Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
            if (J)
                J->coeffRef(0, 0) = 1;
            return Vec::Constant(1, x[0] - 2);
        }
    } r;
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-8;
    auto result = gauss_newton_solve(r, Vec::Zero(1), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x[0], 2, 1e-8);
}

TEST(GaussNewton, SparseRectangularProblem) {
    constexpr int n = 2000;
    SpMat pattern(2 * n, n);
    pattern.reserve(2 * n);
    for (int j = 0; j < n; ++j) {
        pattern.insert(2 * j, j) = 1;
        pattern.insert(2 * j + 1, j) = 1;
    }
    pattern.makeCompressed();
    TestResponse residual(pattern, [](const ConstVecRef& x, SpMat* J) -> Vec {
        Vec r(2 * x.size());
        for (int j = 0; j < x.size(); ++j) {
            r[2 * j] = x[j] - 1;
            r[2 * j + 1] = 2 * (x[j] - 1);
            if (J) {
                J->coeffRef(2 * j, j) = 1;
                J->coeffRef(2 * j + 1, j) = 2;
            }
        }
        return r;
    });
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    auto result = gauss_newton_solve(residual, Vec::Zero(n), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_LT((result.x.array() - 1).abs().maxCoeff(), 1e-9);
}

TEST(GaussNewton, PatternChangesDuringSolve) {
    class Piecewise final : public energy::DynamicPatternResponse {
      public:
        Piecewise() : DynamicPatternResponse(2, 2) {}
        mutable int changes = 0;

      private:
        SpMat jacobian_pattern_at(const ConstVecRef& x) const override {
            SpMat p(2, 2);
            p.insert(0, 0) = 0;
            p.insert(1, 1) = 0;
            if (x[0] > 0)
                p.insert(1, 0) = 0;
            p.makeCompressed();
            ++changes;
            return p;
        }
        Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
            const double positive = std::max(0., x[0]);
            if (J) {
                J->coeffRef(0, 0) = 1;
                J->coeffRef(1, 1) = 1;
                if (x[0] > 0)
                    J->coeffRef(1, 0) = 2 * x[0];
            }
            return (Vec(2) << x[0] - 1, x[1] + positive * positive - 2).finished();
        }
    } residual;
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    auto result = gauss_newton_solve(residual, Vec::Constant(2, -1), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_LT((result.x - Vec::Ones(2)).norm(), 1e-8);
    EXPECT_EQ(residual.eval_jacobian(result.x).nonZeros(), 3);
    EXPECT_GT(residual.changes, 2);
}

TEST(GaussNewton, ReusesAnalysisAndRetriesFactorization) {
    class CountingSolver final : public linalg::SparseSolver {
      public:
        int analyses = 0, factorizations = 0;

      private:
        linalg::EigenLDLT backend;
        bool analyze_pattern(const SpMat& A) override {
            ++analyses;
            return backend.analyze(A);
        }
        bool factor_values(const SpMat& A) override {
            ++factorizations;
            if (factorizations == 1)
                return false;
            return backend.factorize(A);
        }
        void solve_factored(const ConstVecRef& b, VecRef x) const override {
            backend.solve(b, x);
        }
    } backend;
    auto residual = make_response(2, 2, [](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        if (J)
            *J = Eigen::Matrix2d::Identity();
        return x - Vec::Ones(2);
    });
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    auto result = gauss_newton_solve(residual, Vec::Zero(2), options, backend);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_EQ(backend.analyses, 1);
    EXPECT_GT(backend.factorizations, result.iterations);
    EXPECT_LT((result.x - Vec::Ones(2)).norm(), 1e-8);
}

TEST(GaussNewton, NearlyDependentColumns) {
    auto residual = make_response(3, 2, [](const ConstVecRef& x, Eigen::MatrixXd* J) -> Vec {
        Eigen::Matrix<double, 3, 2> A;
        A << 1, 1, 1, 1.001, 1, .999;
        if (J)
            *J = A;
        return A * (x - Eigen::Vector2d(1, 2));
    });
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-11;
    auto result = gauss_newton_solve(residual, Vec::Zero(2), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_LT((result.x - Eigen::Vector2d(1, 2)).norm(), 1e-4);
}

TEST(GaussNewton, DoesNotPrepareUnusedHessianPattern) {
    class Residual final : public energy::DynamicPatternResponse {
      public:
        Residual() : DynamicPatternResponse(1, 1) {}
        bool has_hessian() const override {
            return true;
        }

      private:
        SpMat jacobian_pattern_at(const ConstVecRef&) const override {
            return Eigen::MatrixXd::Identity(1, 1).sparseView();
        }
        SpMat hessian_pattern_at(const ConstVecRef&) const override {
            throw std::logic_error("Gauss-Newton must not request Hessian structure");
        }
        Vec eval_impl(const ConstVecRef& x, SpMat* J) override {
            if (J)
                J->coeffs().setOnes();
            return x - Vec::Ones(1);
        }
    } residual;
    GaussNewtonOptions options;
    options.gradient_tolerance = 1e-9;
    const auto result = gauss_newton_solve(residual, Vec::Zero(1), options);
    EXPECT_EQ(result.status, GaussNewtonStatus::Converged);
    EXPECT_NEAR(result.x[0], 1, 1e-8);
}
