#include "litepgo/linalg/auto_sparse_solver.h"
#include "litepgo/linalg/eigen_ldlt.h"
#include "litepgo/linalg/eigen_sparse_lu.h"
#include "litepgo/linalg/sparse_solver.h"

#if defined(LITEPGO_HAS_ACCELERATE)
#include "litepgo/linalg/accelerate_ldlt.h"
#include "litepgo/linalg/accelerate_llt.h"
#endif

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using litepgo::SpMat;
using litepgo::Triplet;
using litepgo::Vec;

// 1D Laplacian with Dirichlet ends, scaled: symmetric positive definite.
SpMat laplacian(int n, double scale = 1.0) {
    std::vector<Triplet> entries;
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(i, i, 2.0 * scale);
        if (i + 1 < n) {
            entries.emplace_back(i, i + 1, -scale);
            entries.emplace_back(i + 1, i, -scale);
        }
    }
    SpMat A(n, n);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

// Saddle-point matrix [[K, B^T], [B, 0]] with K a Laplacian and B a row of ones:
// symmetric, nonsingular, indefinite.
SpMat saddle_point(int n) {
    std::vector<Triplet> entries;
    const SpMat K = laplacian(n);
    for (int column = 0; column < K.outerSize(); ++column) {
        for (SpMat::InnerIterator it(K, column); it; ++it) {
            entries.emplace_back(it.row(), it.col(), it.value());
        }
    }
    for (int i = 0; i < n; ++i) {
        entries.emplace_back(n, i, 1.0);
        entries.emplace_back(i, n, 1.0);
    }
    SpMat A(n + 1, n + 1);
    A.setFromTriplets(entries.begin(), entries.end());
    return A;
}

Vec dense_solution(const SpMat& A, const Vec& rhs) {
    return Eigen::MatrixXd(A).partialPivLu().solve(rhs);
}

template <class Solver> void expect_solves(Solver& solver, const SpMat& A) {
    const Vec rhs = Vec::LinSpaced(A.rows(), -1.0, 2.0);
    ASSERT_TRUE(solver.factorize(A));
    EXPECT_LT((solver.solve(rhs) - dense_solution(A, rhs)).norm(), 1e-10);
}

// A test backend over Eigen's LDLT that counts analyses and can be told to fail.
class CountingSolver final : public litepgo::SparseSolver {
  public:
    int analyze_calls = 0;
    bool fail_next_factor = false;
    bool solve_inside_factor = false;

  private:
    void analyze(const SpMat& A) override {
        ++analyze_calls;
        solver_.analyzePattern(A);
    }
    bool factor(const SpMat& A) override {
        if (solve_inside_factor) {
            solve(Vec::Zero(A.rows()));
        }
        if (fail_next_factor) {
            fail_next_factor = false;
            return false;
        }
        solver_.factorize(A);
        return solver_.info() == Eigen::Success;
    }
    void solve_factored(const litepgo::ConstVecRef& rhs, litepgo::VecRef x) const override {
        x = solver_.solve(rhs);
    }

    Eigen::SimplicialLDLT<SpMat> solver_;
};

} // namespace

// Every backend on a positive definite matrix.
template <class Solver> class AllSolvers : public ::testing::Test {};
using AllSolverTypes = ::testing::Types<litepgo::EigenLDLT, litepgo::EigenSparseLU
#if defined(LITEPGO_HAS_ACCELERATE)
                                        ,
                                        litepgo::AccelerateLLT, litepgo::AccelerateLDLT
#endif
                                        >;
TYPED_TEST_SUITE(AllSolvers, AllSolverTypes);

TYPED_TEST(AllSolvers, SolvesPositiveDefinite) {
    TypeParam solver;
    expect_solves(solver, laplacian(40));
}

TYPED_TEST(AllSolvers, RefactorsNewValuesOnTheSamePattern) {
    TypeParam solver;
    expect_solves(solver, laplacian(40));
    expect_solves(solver, laplacian(40, 3.5));
}

TYPED_TEST(AllSolvers, RefactorsAChangedPattern) {
    TypeParam solver;
    expect_solves(solver, laplacian(40));
    expect_solves(solver, laplacian(25));
}

TYPED_TEST(AllSolvers, SolvesIntoProvidedVector) {
    TypeParam solver;
    const SpMat A = laplacian(10);
    const Vec rhs = Vec::Ones(10);
    ASSERT_TRUE(solver.factorize(A));
    Vec x = Vec::Zero(10);
    solver.solve(rhs, x);
    EXPECT_LT((x - dense_solution(A, rhs)).norm(), 1e-10);
}

// Backends that handle symmetric indefinite matrices.
template <class Solver> class IndefiniteSolvers : public ::testing::Test {};
using IndefiniteSolverTypes = ::testing::Types<litepgo::EigenSparseLU
#if defined(LITEPGO_HAS_ACCELERATE)
                                               ,
                                               litepgo::AccelerateLDLT
#endif
                                               >;
TYPED_TEST_SUITE(IndefiniteSolvers, IndefiniteSolverTypes);

TYPED_TEST(IndefiniteSolvers, SolvesSaddlePoint) {
    TypeParam solver;
    expect_solves(solver, saddle_point(12));
}

#if defined(LITEPGO_HAS_ACCELERATE)
TEST(AccelerateLLT, FailsOnIndefiniteAndDropsTheFactorization) {
    litepgo::AccelerateLLT solver;
    EXPECT_FALSE(solver.factorize(saddle_point(12)));
    EXPECT_THROW(solver.solve(Vec::Ones(13)), std::logic_error);
    expect_solves(solver, laplacian(12));
}
#endif

TEST(AutoSparseSolver, AliasesThePlatformDefaults) {
#if defined(LITEPGO_HAS_ACCELERATE)
    EXPECT_TRUE((std::is_same_v<litepgo::AutoSparseSolver, litepgo::AccelerateLLT>));
    EXPECT_TRUE((std::is_same_v<litepgo::AutoIndefiniteSparseSolver, litepgo::AccelerateLDLT>));
#else
    EXPECT_TRUE((std::is_same_v<litepgo::AutoSparseSolver, litepgo::EigenLDLT>));
    EXPECT_TRUE((std::is_same_v<litepgo::AutoIndefiniteSparseSolver, litepgo::EigenSparseLU>));
#endif
}

TEST(SparseSolver, ReusesTheAnalysisWhileThePatternHolds) {
    CountingSolver solver;
    ASSERT_TRUE(solver.factorize(laplacian(20)));
    ASSERT_TRUE(solver.factorize(laplacian(20, 2.0)));
    EXPECT_EQ(solver.analyze_calls, 1);
    ASSERT_TRUE(solver.factorize(laplacian(21)));
    EXPECT_EQ(solver.analyze_calls, 2);
}

TEST(SparseSolver, FailureDropsTheFactorizationAndTheAnalysis) {
    CountingSolver solver;
    ASSERT_TRUE(solver.factorize(laplacian(20)));
    solver.fail_next_factor = true;
    EXPECT_FALSE(solver.factorize(laplacian(20)));
    EXPECT_EQ(solver.analyze_calls, 1); // the failed call still matched the pattern
    EXPECT_THROW(solver.solve(Vec::Ones(20)), std::logic_error);
    ASSERT_TRUE(solver.factorize(laplacian(20)));
    EXPECT_EQ(solver.analyze_calls, 2); // but the next call analyzes again
}

TEST(SparseSolver, RejectsInvalidMatrices) {
    litepgo::EigenLDLT solver;
    EXPECT_THROW(solver.factorize(SpMat(0, 0)), std::invalid_argument);
    EXPECT_THROW(solver.factorize(SpMat(3, 4)), std::invalid_argument);

    SpMat uncompressed(3, 3);
    uncompressed.reserve(Eigen::VectorXi::Constant(3, 1));
    for (int i = 0; i < 3; ++i) {
        uncompressed.insert(i, i) = 1.0;
    }
    ASSERT_FALSE(uncompressed.isCompressed());
    EXPECT_THROW(solver.factorize(uncompressed), std::invalid_argument);

    // Eigen asserts when it copies such matrices, so edit the row indices in place.
    // Column 0 of laplacian(2) holds rows {0, 1}.
    SpMat duplicated = laplacian(2);
    duplicated.innerIndexPtr()[1] = 0;
    EXPECT_THROW(solver.factorize(duplicated), std::invalid_argument);
    SpMat unsorted = laplacian(2);
    std::swap(unsorted.innerIndexPtr()[0], unsorted.innerIndexPtr()[1]);
    EXPECT_THROW(solver.factorize(unsorted), std::invalid_argument);
}

TEST(SparseSolver, ChecksSolveArguments) {
    litepgo::EigenLDLT solver;
    EXPECT_THROW(solver.solve(Vec::Ones(5)), std::logic_error);
    ASSERT_TRUE(solver.factorize(laplacian(5)));
    EXPECT_THROW(solver.solve(Vec::Ones(4)), std::invalid_argument);
    Vec x = Vec::Ones(5);
    EXPECT_THROW(solver.solve(x, x), std::invalid_argument);
}

TEST(SparseSolver, RejectsOverlappingUse) {
    CountingSolver solver;
    solver.solve_inside_factor = true;
    EXPECT_THROW(solver.factorize(laplacian(5)), std::logic_error);
    solver.solve_inside_factor = false;
    expect_solves(solver, laplacian(5));
}
